import copy
from itertools import repeat

import numpy as np

from openmc.mpi import comm
from openmc.utility_funcs import change_directory

from .abc import Integrator, SIIntegrator, OperatorResult, add_params
from ._matrix_funcs import (
    cf4_f1, cf4_f2, cf4_f3, cf4_f4, celi_f1, celi_f2,
    leqi_f1, leqi_f2, leqi_f3, leqi_f4, rk4_f1, rk4_f4
)

__all__ = [
    "PredictorIntegrator", "CECMIntegrator", "CF4Integrator",
    "CELIIntegrator", "EPCRK4Integrator", "LEQIIntegrator",
    "SICELIIntegrator", "SILEQIIntegrator"]


@add_params
class PredictorIntegrator(Integrator):
    r"""Deplete using a first-order predictor algorithm.

    Implements the first-order predictor algorithm. This algorithm is
    mathematically defined as:

    .. math::
        \mathbf{n}_{i+1} = \exp\left(h\mathbf{A}(\mathbf{n}_i) \right) \mathbf{n}_i
    """
    _num_stages = 1

    def __call__(self, n, rates, dt, source_rate, _i=None):
        """Perform the integration across one time step

        Parameters
        ----------
        n : list of numpy.ndarray
            List of atom number arrays for each material. Each array in the list
            contains the number of [atom] of each nuclide.
        rates : openmc.deplete.ReactionRates
            Reaction rates from operator
        dt : float
            Time in [s] for the entire depletion interval
        source_rate : float
            Power in [W] or source rate in [neutron/sec]
        _i : int, optional
            Current iteration count. Not used

        Returns
        -------
        proc_time : float
            Time spent in CRAM routines for all materials in [s]
        n_end : list of numpy.ndarray
            Concentrations at end of interval

        """
        proc_time, n_end = self._timed_deplete(n, rates, dt, _i)
        return proc_time, n_end


@add_params
class AdaptiveIntegrator(PredictorIntegrator):
    r"""Deplete using a first-order predictor algorithm with adaptive transport.

    This integrator extends :class:`PredictorIntegrator` by adaptively deciding
    when to call the transport operator based on a neutronically weighted
    inventory deviation (NWID) metric. The NWID metric compares the current
    nuclide inventories to those at the last transport solve using
    flux-dependent weighting factors derived from reaction rates.

    At each depletion step, the integrator:

    1. Evaluates the NWID with respect to the last transport point.
    2. If the NWID exceeds a user-specified tolerance, it performs a new
       transport solve and refreshes the weighting factors and reference
       inventories.
    3. Otherwise, it reuses the reaction rates from the last transport solve
       for the depletion step.

    The weighting factors for each nuclide are proportional to a power
    of its absorption reaction rate per atom at the last transport point,
    :math:`w_{m,\text{nuc}} \propto r_{\text{abs}}^\alpha`, where
    :math:`r_{\text{abs}}` has units of [(reactions/s)/atom] and
    :math:`0 < \alpha < 1`.

    The global NWID metric used for triggering new transport solves is

    .. math::

        \text{NWID} = 100 \cdot
        \frac{\sum_{m,\text{nuc}} w_{m,\text{nuc}}
              \left|\;N_{m,\text{nuc}}(t) - N_{m,\text{nuc}}(t_0)\;\right|}
             {\sum_{m,\text{nuc}} w_{m,\text{nuc}} N_{m,\text{nuc}}(t_0)},

    where :math:`t_0` corresponds to the most recent transport solve and the
    sum runs over all monitored materials and nuclides.
    """

    _num_stages = 1

    def __init__(
        self,
        operator,
        timesteps,
        power=None,
        power_density=None,
        source_rates=None,
        timestep_units: str = "s",
        solver: str = "cram48",
        continue_timesteps: bool = False,
        nwid_tolerance: float = 5.0,
        alpha: float = 0.75,
        materials=None,
        weight_reactions=('(n,gamma)', '(n,2n)', '(n,p)', '(n,a)', '(n,3n)', '(n,4n)', 'fission')
    ):
        """Construct an adaptive predictor integrator.

        Parameters
        ----------
        operator : openmc.deplete.abc.TransportOperator
            Transport operator used to generate reaction rates. For adaptive
            transport with spectral feedback, this is expected to be an
            instance of :class:`openmc.deplete.CoupledOperator`.
        timesteps : iterable of float or iterable of tuple
            Depletion time steps, as in :class:`Integrator`.
        power : float or iterable of float, optional
            Reactor power in [W].
        power_density : float or iterable of float, optional
            Reactor power density in [W/gHM].
        source_rates : float or iterable of float, optional
            Source rate in [neutron/sec] or [neutron/s-cm^2].
        timestep_units : {'s', 'min', 'h', 'd', 'a', 'MWd/kg'}
            Units for values specified in ``timesteps``.
        solver : str or callable, optional
            Bateman solver, as in :class:`Integrator`.
        continue_timesteps : bool, optional
            Continuation from previous results. Not currently supported for the
            adaptive integrator.
        nwid_tolerance : float, optional
            NWID tolerance in percent. A new transport solve is triggered when
            the NWID between the current and reference inventories exceeds
            this value.
        alpha : float, optional
            Exponent used when forming weighting factors from absorption
            reaction rates per atom.
        materials : iterable, optional
            Optional iterable of material IDs or :class:`openmc.Material`
            instances to include in the NWID metric. If omitted, all
            depletable materials handled by the operator are included.
        weight_reactions : iterable of str, optional
            Names of reactions to include when forming weighting factors.
            The default is ``("absorption",)``. When multiple reactions are
            provided, their reaction rates per atom are summed before applying
            the power :math:`\\alpha`.
        """
        if continue_timesteps:
            raise ValueError(
                "AdaptiveIntegrator does not currently support continue_timesteps. "
                "Restart calculations should be performed with a non-adaptive integrator."
            )
        if getattr(operator, "prev_res", None) is not None:
            raise ValueError(
                "AdaptiveIntegrator does not currently support prev_results. "
                "Restart calculations should be performed with a non-adaptive integrator."
            )

        super().__init__(
            operator,
            timesteps,
            power=power,
            power_density=power_density,
            source_rates=source_rates,
            timestep_units=timestep_units,
            solver=solver,
            continue_timesteps=False,
        )

        if nwid_tolerance <= 0.0:
            raise ValueError("nwid_tolerance must be positive.")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")

        self.nwid_tolerance = float(nwid_tolerance)
        self.alpha = float(alpha)

        # Normalize reaction names used in the weighting factors
        if weight_reactions is None:
            self._weight_reactions = ("absorption",)
        elif isinstance(weight_reactions, str):
            self._weight_reactions = (weight_reactions,)
        else:
            self._weight_reactions = tuple(weight_reactions)

        # Optional material filter for NWID metric: store material IDs as strings
        if materials is None:
            self._material_filter = None
        else:
            mat_ids = []
            for mat in materials:
                if hasattr(mat, "id"):
                    mat_ids.append(str(mat.id))
                else:
                    mat_ids.append(str(mat))
            self._material_filter = set(mat_ids)

        # Internal state for NWID metric
        self._nwid_material_ids = None           # ordered list of material IDs (strings)
        self._material_index_map = None          # mat_id -> local index in composition vectors
        self._monitored_nuclides = None          # ordered list of nuclide names used in NWID
        self._nuc_index_in_number = None         # nuclide -> index in AtomNumber / composition vectors
        self._reference_number = {}              # mat_id -> reference inventory vector
        self._weights = {}                       # mat_id -> weighting factor vector
        self._nwid_denominator = 0.0             # global denominator for NWID normalization

        # Diagnostics for adaptive transport placement
        self._transport_step_indices = []        # depletion step indices where transport ran
        self.transport_times = []                # physical times where transport ran
        self.transport_keff = []                 # (k, sigma) at each transport time

        # Source rate at the last transport solve. Used in fallback
        # normalization when per-source reaction rates are not available.
        self._last_transport_source_rate = None

        # Per-source reaction rates from the last transport solve in units of
        # [(reactions/src)/atom]. When available, these are used to recompute
        # the normalization to power or source rate for new compositions
        # between transport solves.
        self._per_source_rates = None

    def _initialize_nwid_materials(self):
        """Determine which materials contribute to the NWID metric.

        This uses the operator's ``local_mats`` ordering, which is consistent
        with the composition vectors produced by ``initial_condition`` and the
        material dimension of :class:`~openmc.deplete.ReactionRates`.
        """
        local_mats = getattr(self.operator, "local_mats", None)
        if local_mats is None:
            raise AttributeError(
                "AdaptiveIntegrator requires an operator with a 'local_mats' "
                "attribute (e.g., openmc.deplete.CoupledOperator)."
            )

        material_ids = []
        material_index_map = {}
        for idx, mat_id in enumerate(local_mats):
            if self._material_filter is not None and mat_id not in self._material_filter:
                continue
            material_ids.append(mat_id)
            material_index_map[mat_id] = idx

        if not material_ids:
            raise ValueError(
                "No depletable materials were selected for the NWID metric. "
                "Check the materials passed to AdaptiveIntegrator."
            )

        self._nwid_material_ids = material_ids
        self._material_index_map = material_index_map

    def _initialize_nwid_nuclides(self, rates):
        """Initialize nuclide indexing used by the NWID metric.

        The NWID metric is constructed over the nuclides that appear in the
        operator's reaction rate tallies and have corresponding entries in the
        AtomNumber object used to build composition vectors.
        """
        if self._monitored_nuclides is not None:
            return

        monitored_nucs = list(rates.index_nuc.keys())
        if not monitored_nucs:
            raise ValueError(
                "ReactionRates object does not contain any nuclides. "
                "Cannot construct NWID weighting factors."
            )

        # Map nuclide names to indices in the composition vectors returned by
        # AtomNumber. The order of these vectors is given by
        # self.operator.number.nuclides.
        number_nuclides = list(getattr(self.operator.number, "nuclides", []))
        index_in_number = {}
        for nuc in monitored_nucs:
            try:
                index_in_number[nuc] = number_nuclides.index(nuc)
            except ValueError:
                # Skip nuclides that are present in reaction rate tallies but
                # not in the AtomNumber indexing.
                continue

        if not index_in_number:
            raise ValueError(
                "No overlap between nuclides in reaction-rate tallies and "
                "the composition vectors used for depletion."
            )

        self._monitored_nuclides = monitored_nucs
        self._nuc_index_in_number = index_in_number

    def _update_nwid_reference(self, n_bos, rates, step_index, current_time, output):
        """Update reference inventories and weighting factors after transport.

        This method selects the inventories and reaction-rate-based weighting
        factors that define the NWID reference state at the most recent
        transport solve.
        """
        if self._nwid_material_ids is None or self._material_index_map is None:
            self._initialize_nwid_materials()
        self._initialize_nwid_nuclides(rates)

        reference_number = {}
        weights = {}
        denominator = 0.0

        # Pre-compute indices for the reactions that contribute to the weights
        rx_indices = []
        for rx in self._weight_reactions:
            idx = rates.index_rx.get(rx)
            if idx is not None:
                rx_indices.append(idx)
        if not rx_indices:
            raise ValueError(
                "None of the requested weight_reactions are present in the "
                "ReactionRates index. Available reactions are: "
                f"{list(rates.index_rx.keys())}"
            )

        for mat_id in self._nwid_material_ids:
            local_idx = self._material_index_map[mat_id]
            rr_mat_idx = rates.index_mat.get(mat_id)
            if rr_mat_idx is None:
                # If for some reason reaction rates are missing for this
                # material, skip it in the NWID metric.
                continue

            n0_vec = np.zeros(len(self._monitored_nuclides))
            w_vec = np.zeros(len(self._monitored_nuclides))

            for j, nuc in enumerate(self._monitored_nuclides):
                inv_idx = self._nuc_index_in_number.get(nuc)
                rr_nuc_idx = rates.index_nuc.get(nuc)
                if inv_idx is None or rr_nuc_idx is None:
                    continue

                n0 = n_bos[local_idx][inv_idx]
                n0_vec[j] = n0
                if n0 <= 0.0:
                    continue

                # Sum the requested reaction rates per atom for this nuclide
                # in this material at the reference step.
                rate_value = 0.0
                for rx_idx in rx_indices:
                    rate_value += rates[rr_mat_idx, rr_nuc_idx, rx_idx]

                if rate_value <= 0.0:
                    continue

                w_vec[j] = rate_value ** self.alpha

            reference_number[mat_id] = n0_vec
            weights[mat_id] = w_vec
            denominator += np.sum(w_vec * n0_vec)

        if denominator <= 0.0:
            raise ValueError(
                "AdaptiveIntegrator constructed a zero NWID denominator. "
                "This typically indicates that all monitored nuclides have "
                "zero inventories or zero reaction rates in the reference "
                "state. Consider adjusting the monitored materials or "
                "weight_reactions."
            )

        self._reference_number = reference_number
        self._weights = weights
        self._nwid_denominator = float(denominator)

        # Record this transport event for diagnostics
        self._transport_step_indices.append(step_index)
        self.transport_times.append(current_time)

        # Store keff and its uncertainty at this transport point
        if rates is not None:
            # The corresponding OperatorResult is expected to have been
            # generated immediately before calling this method.
            # transport_keff is appended in integrate() when the OperatorResult
            # is available.
            pass

        # Optionally print the weighting factors for diagnostics on rank 0
        if output and comm.rank == 0:
            print(
                "[openmc.deplete] AdaptiveIntegrator: updated NWID reference "
                f"at step {step_index} (t={current_time} s)."
            )
            print("[openmc.deplete] AdaptiveIntegrator weighting factors:")
            for mat_id in self._nwid_material_ids:
                w_vec = self._weights.get(mat_id)
                if w_vec is None or not np.any(w_vec):
                    continue
                print(f"  material {mat_id}:")
                for nuc, w_val in zip(self._monitored_nuclides, w_vec):
                    if w_val <= 0.0:
                        continue
                    print(f"    {nuc}: {w_val:.6e}")

    def _compute_nwid(self, n_bos):
        """Compute the current NWID percentage with respect to the reference."""
        if self._nwid_denominator <= 0.0:
            return 0.0

        numerator = 0.0
        for mat_id in self._nwid_material_ids:
            local_idx = self._material_index_map[mat_id]
            n0_vec = self._reference_number.get(mat_id)
            w_vec = self._weights.get(mat_id)
            if n0_vec is None or w_vec is None or not np.any(w_vec):
                continue

            n_vec = np.zeros_like(n0_vec)
            for j, nuc in enumerate(self._monitored_nuclides):
                inv_idx = self._nuc_index_in_number.get(nuc)
                if inv_idx is None:
                    continue
                n_vec[j] = n_bos[local_idx][inv_idx]

            diff = np.abs(n_vec - n0_vec)
            numerator += np.sum(w_vec * diff)

        if numerator <= 0.0:
            return 0.0
        return 100.0 * float(numerator) / self._nwid_denominator

    def _recompute_rates_from_per_source(
        self,
        n_bos,
        per_source_rates,
        source_rate: float,
    ):
        """Recompute normalized reaction rates from per-source rates.

        This helper reconstructs the normalization factor used for power-based
        depletion (``fission-q`` mode) given:

        * the per-source reaction rates in units of [(reactions/src)/atom],
        * the current beginning-of-step inventories, and
        * the requested source rate / power for this depletion interval.

        When successful, it returns a new :class:`ReactionRates` object with
        units of [(reactions/sec)/atom] that is consistent with the current
        composition but retains the spectrum and microscopic information from
        the last transport solve.
        """
        # Only support recomputation when the operator uses fission-q
        # normalization. For other normalization schemes, fall back to simple
        # scaling in the caller.
        normalization_helper = getattr(self.operator, "_normalization_helper", None)
        from .helpers import ChainFissionHelper

        if not isinstance(normalization_helper, ChainFissionHelper):
            raise RuntimeError(
                "AdaptiveIntegrator per-source normalization is only supported "
                "for operators using fission-q normalization."
            )

        # Ensure NWID indexing structures are initialized so that we can map
        # between nuclides in the per-source reaction rates and indices in the
        # composition vectors.
        if self._nwid_material_ids is None or self._material_index_map is None:
            self._initialize_nwid_materials()
        self._initialize_nwid_nuclides(per_source_rates)

        # Build a helper with an energy vector consistent with the depletion
        # chain and the nuclide ordering used in the per-source reaction rates.
        helper = ChainFissionHelper()
        helper.prepare(self.operator.chain.nuclides, per_source_rates.index_nuc)
        helper.reset()

        fission_idx = per_source_rates.index_rx.get("fission")
        if fission_idx is None:
            # No fission reaction present in the tallies; treat the per-source
            # rates as proportional to source rate only.
            return per_source_rates * source_rate

        # Accumulate fission rates per source neutron for each material and
        # update the helper with the corresponding energy production.
        for mat_id in self._nwid_material_ids:
            local_idx = self._material_index_map[mat_id]
            rr_mat_idx = per_source_rates.index_mat.get(mat_id)
            if rr_mat_idx is None:
                continue

            fission_rates = np.zeros(len(per_source_rates.index_nuc))
            for nuc, rr_nuc_idx in per_source_rates.index_nuc.items():
                inv_idx = self._nuc_index_in_number.get(nuc)
                if inv_idx is None:
                    continue
                n_atoms = n_bos[local_idx][inv_idx]
                if n_atoms <= 0.0:
                    continue

                rate_per_atom_src = per_source_rates[rr_mat_idx, rr_nuc_idx, fission_idx]
                if rate_per_atom_src <= 0.0:
                    continue

                # Convert per-atom per-source reaction rate to per-source rate
                # by multiplying by the total number of atoms.
                fission_rates[rr_nuc_idx] = rate_per_atom_src * n_atoms

            helper.update(fission_rates)

        # Compute the normalization factor for this step and apply it to all
        # per-source reaction rates to obtain [(reactions/sec)/atom].
        factor = helper.factor(source_rate)
        return per_source_rates * factor

    def integrate(
        self,
        final_step: bool = True,
        output: bool = True,
        path: str = "depletion_results.h5",
        write_rates: bool = False,
    ):
        """Perform adaptive depletion across all steps.

        This method mirrors :meth:`Integrator.integrate` but replaces the
        one-transport-per-step pattern with an adaptive strategy that may skip
        transport solves when the NWID metric is below the specified tolerance.
        """
        import h5py

        with change_directory(self.operator.output_dir):
            n = self.operator.initial_condition()
            t, self._i_res = self._get_start_data()

            last_res = None

            for i, (dt, source_rate) in enumerate(self):
                if output and comm.rank == 0:
                    print(
                        f"[openmc.deplete] t={t} s, dt={dt} s, "
                        f"source={source_rate}"
                    )

                # For the very first step, or when there is no reference yet,
                # always perform a transport solve to establish the NWID
                # baseline.
                perform_transport = False
                if i == 0 or self._nwid_denominator <= 0.0 or last_res is None:
                    perform_transport = True
                else:
                    # Evaluate NWID at the current beginning-of-step
                    nwid_value = self._compute_nwid(n)
                    if output and comm.rank == 0:
                        print(
                            "[openmc.deplete] AdaptiveIntegrator: "
                            f"NWID={nwid_value:.6f}% (tolerance "
                            f"{self.nwid_tolerance:.6f}%)"
                        )
                    if nwid_value >= self.nwid_tolerance:
                        perform_transport = True

                if perform_transport:
                    # Obtain BOS concentrations and reaction rates from the
                    # operator. This also writes BOS statepoint data.
                    n, res = self._get_bos_data_from_operator(i, source_rate, n)
                    last_res = res
                    self._last_transport_source_rate = source_rate

                    # Cache per-source reaction rates from the operator, if
                    # available. These are used to recompute the normalization
                    # for new compositions between transport solves without
                    # performing an additional transport calculation.
                    self._per_source_rates = getattr(
                        self.operator, "_rates_per_source", None
                    )

                    # Record keff diagnostics for this transport point.
                    self.transport_keff.append(
                        (res.k.nominal_value, res.k.std_dev)
                    )

                    # Update reference inventories and weighting factors used
                    # for the NWID metric.
                    self._update_nwid_reference(
                        n,
                        res.rates,
                        self._i_res + i,
                        t,
                        output,
                    )
                else:
                    # Reuse information from the last transport solve. When
                    # per-source reaction rates are available and the operator
                    # uses fission-q normalization, recompute the normalization
                    # factor for the current composition using those
                    # per-source rates. This reproduces the behavior of a
                    # one-transport / many-depletion-steps scheme while
                    # retaining continuous-energy reaction-rate fidelity.
                    if self._per_source_rates is not None:
                        try:
                            step_rates = self._recompute_rates_from_per_source(
                                n, self._per_source_rates, source_rate
                            )
                            res = OperatorResult(last_res.k, step_rates)
                        except Exception:
                            # Fall back to simple scaling if recomputation
                            # fails for any reason.
                            res = None
                    else:
                        res = None

                    if res is None:
                        # Fallback: scale reaction rates by the ratio of the
                        # current source rate to the source rate at the last
                        # transport, matching the restart behavior in the base
                        # Integrator.
                        if (
                            self._last_transport_source_rate is not None
                            and self._last_transport_source_rate != 0.0
                        ):
                            scaled_rates = last_res.rates.copy()
                            scaled_rates *= (
                                source_rate / self._last_transport_source_rate
                            )
                            res = OperatorResult(last_res.k, scaled_rates)
                        else:
                            res = last_res

                    # Keep the operator's internal composition (operator.number)
                    # in sync with the current BOS so any code that reads it
                    # sees the correct state. We do not run transport here, so
                    # OpenMC C-side materials are not updated until the next
                    # transport.
                    if hasattr(self.operator, "number") and hasattr(
                        self.operator.number, "set_density"
                    ):
                        self.operator.number.set_density(n)

                # Deplete across the interval using the chosen reaction rates.
                proc_time, n_end = self(n, res.rates, dt, source_rate, i)

                from .stepresult import StepResult

                StepResult.save(
                    self.operator,
                    n,
                    res,
                    [t, t + dt],
                    source_rate,
                    self._i_res + i,
                    proc_time,
                    write_rates=write_rates,
                    path=path,
                )

                n = n_end
                t += dt

            # Optional final operator evaluation at end of history.
            if output and final_step and comm.rank == 0:
                print(
                    f"[openmc.deplete] t={t} "
                    "(final operator evaluation for AdaptiveIntegrator)"
                )
            res_final = self.operator(n, source_rate if final_step else 0.0)

            # Record diagnostics for the final transport evaluation, if any.
            if final_step:
                self.transport_keff.append(
                    (res_final.k.nominal_value, res_final.k.std_dev)
                )
                self._transport_step_indices.append(self._i_res + len(self))
                self.transport_times.append(t)

            from .stepresult import StepResult

            StepResult.save(
                self.operator,
                n,
                res_final,
                [t, t],
                source_rate,
                self._i_res + len(self),
                None,
                write_rates=write_rates,
                path=path,
            )
            self.operator.write_bos_data(len(self) + self._i_res)

            # Persist adaptive diagnostics in the depletion results file so that
            # they can be interrogated via the Results API or direct HDF5
            # access.
            if comm.rank == 0:
                transport_indices = np.asarray(
                    self._transport_step_indices, dtype=int
                )
                transport_times = np.asarray(
                    self.transport_times, dtype=float
                )
                transport_keff = (
                    np.asarray(self.transport_keff, dtype=float)
                    if self.transport_keff
                    else np.zeros((0, 2), dtype=float)
                )

                with h5py.File(path, "a") as fh:
                    grp = fh.require_group("adaptive")

                    for name in (
                        "transport_step_indices",
                        "transport_times",
                        "transport_keff",
                    ):
                        if name in grp:
                            del grp[name]

                    grp.create_dataset(
                        "transport_step_indices",
                        data=transport_indices,
                    )
                    grp.create_dataset(
                        "transport_times",
                        data=transport_times,
                    )
                    grp.create_dataset(
                        "transport_keff",
                        data=transport_keff,
                    )

        self.operator.finalize()


@add_params
class CECMIntegrator(Integrator):
    r"""Deplete using the CE/CM algorithm.

    Implements the second order `CE/CM predictor-corrector algorithm
    <https://doi.org/10.13182/NSE14-92>`_.

    "CE/CM" stands for constant extrapolation on predictor and constant
    midpoint on corrector. This algorithm is mathematically defined as:

    .. math::
        \begin{aligned}
        \mathbf{n}_{i+1/2} &= \exp \left (\frac{h}{2}\mathbf{A}(\mathbf{n}_i)
            \right) \mathbf{n}_i \\
        \mathbf{n}_{i+1} &= \exp \left(h \mathbf{A}(\mathbf{n}_{i+1/2}) \right)
            \mathbf{n}_i.
        \end{aligned}
    """
    _num_stages = 2

    def __call__(self, n, rates, dt, source_rate, _i=None):
        """Integrate using CE/CM

        Parameters
        ----------
        n : list of numpy.ndarray
            List of atom number arrays for each material. Each array in the list
            contains the number of [atom] of each nuclide.
        rates : openmc.deplete.ReactionRates
            Reaction rates from operator
        dt : float
            Time in [s] for the entire depletion interval
        source_rate : float
            Power in [W] or source rate in [neutron/sec]
        _i : int, optional
            Current iteration count. Not used

        Returns
        -------
        proc_time : float
            Time spent in CRAM routines for all materials in [s]
        n_end : list of numpy.ndarray
            Concentrations at end of interval
        """
        # deplete across first half of interval
        time0, n_middle = self._timed_deplete(n, rates, dt / 2, _i)
        res_middle = self.operator(n_middle, source_rate)

        # deplete across entire interval with BOS concentrations,
        # MOS reaction rates
        time1, n_end = self._timed_deplete(n, res_middle.rates, dt, _i)

        return time0 + time1, n_end


@add_params
class CF4Integrator(Integrator):
    r"""Deplete using the CF4 algorithm.

    Implements the fourth order `commutator-free Lie algorithm
    <https://doi.org/10.1016/S0167-739X(02)00161-9>`_.
    This algorithm is mathematically defined as:

    .. math::
        \begin{aligned}
        \mathbf{A}_1 &= h\mathbf{A}(\mathbf{n}_i) \\
        \hat{\mathbf{n}}_1 &= \exp \left ( \frac{\mathbf{A}_1}{2} \right ) \mathbf{n}_i \\
        \mathbf{A}_2 &= h\mathbf{A}(\hat{\mathbf{n}}_1) \\
        \hat{\mathbf{n}}_2 &= \exp \left ( \frac{\mathbf{A}_2}{2} \right ) \mathbf{n}_i \\
        \mathbf{A}_3 &= h \mathbf{A}(\hat{\mathbf{n}}_2) \\
        \hat{\mathbf{n}}_3 &= \exp \left ( -\frac{\mathbf{A}_1}{2} + \mathbf{A}_3
            \right ) \hat{\mathbf{n}}_1 \\
        \mathbf{A}_4 &= h\mathbf{A}(\hat{\mathbf{n}}_3) \\
        \mathbf{n}_{i+1} &= \exp \left ( \frac{\mathbf{A}_1}{4} + \frac{\mathbf{A}_2}{6}
            + \frac{\mathbf{A}_3}{6} - \frac{\mathbf{A}_4}{12} \right )
        \exp \left ( -\frac{\mathbf{A}_1}{12} + \frac{\mathbf{A}_2}{6} +
            \frac{\mathbf{A}_3}{6} + \frac{\mathbf{A}_4}{4} \right ) \mathbf{n}_i.
        \end{aligned}
    """
    _num_stages = 4

    def __call__(self, n_bos, bos_rates, dt, source_rate, _i=None):
        """Perform the integration across one time step

        Parameters
        ----------
        n_bos : list of numpy.ndarray
            List of atom number arrays for each material. Each array in the list
            contains the number of [atom] of each nuclide.
        bos_rates : openmc.deplete.ReactionRates
            Reaction rates from operator
        dt : float
            Time in [s] for the entire depletion interval
        source_rate : float
            Power in [W] or source rate in [neutron/sec]
        _i : int, optional
            Current depletion step index. Not used

        Returns
        -------
        proc_time : float
            Time spent in CRAM routines for all materials in [s]
        n_end : list of numpy.ndarray
            Concentrations at end of interval
        """
        # Step 1: deplete with matrix 1/2*A(y0)
        time1, n_eos1 = self._timed_deplete(
            n_bos, bos_rates, dt, _i, matrix_func=cf4_f1)
        res1 = self.operator(n_eos1, source_rate)

        # Step 2: deplete with matrix 1/2*A(y1)
        time2, n_eos2 = self._timed_deplete(
            n_bos, res1.rates, dt, _i, matrix_func=cf4_f1)
        res2 = self.operator(n_eos2, source_rate)

        # Step 3: deplete with matrix -1/2*A(y0)+A(y2)
        list_rates = list(zip(bos_rates, res2.rates))
        time3, n_eos3 = self._timed_deplete(
            n_eos1, list_rates, dt, _i, matrix_func=cf4_f2)
        res3 = self.operator(n_eos3, source_rate)

        # Step 4: deplete with two matrix exponentials
        list_rates = list(zip(bos_rates, res1.rates, res2.rates, res3.rates))
        time4, n_inter = self._timed_deplete(
            n_bos, list_rates, dt, _i, matrix_func=cf4_f3)
        time5, n_eos5 = self._timed_deplete(
            n_inter, list_rates, dt, _i, matrix_func=cf4_f4)

        return time1 + time2 + time3 + time4 + time5, n_eos5


@add_params
class CELIIntegrator(Integrator):
    r"""Deplete using the CE/LI CFQ4 algorithm.

    Implements the CE/LI Predictor-Corrector algorithm using the `fourth order
    commutator-free integrator <https://doi.org/10.1137/05063042>`_.

    "CE/LI" stands for constant extrapolation on predictor and linear
    interpolation on corrector. This algorithm is mathematically defined as:

    .. math::
        \begin{aligned}
        \mathbf{n}_{i+1}^p &= \exp \left ( h \mathbf{A}(\mathbf{n}_i ) \right )
        \mathbf{n}_i \\
        \mathbf{n}_{i+1} &= \exp \left( \frac{h}{12} \mathbf{A}(\mathbf{n}_i) +
            \frac{5h}{12} \mathbf{A}(\mathbf{n}_{i+1}^p) \right)
        \exp \left( \frac{5h}{12} \mathbf{A}(\mathbf{n}_i) +
        \frac{h}{12} \mathbf{A}(\mathbf{n}_{i+1}^p) \right) \mathbf{n}_i.
        \end{aligned}
    """
    _num_stages = 2

    def __call__(self, n_bos, rates, dt, source_rate, _i=None):
        """Perform the integration across one time step

        Parameters
        ----------
        n_bos : list of numpy.ndarray
            List of atom number arrays for each material. Each array in the list
            contains the number of [atom] of each nuclide.
        rates : openmc.deplete.ReactionRates
            Reaction rates from operator
        dt : float
            Time in [s] for the entire depletion interval
        source_rate : float
            Power in [W] or source rate in [neutron/sec]
        _i : int, optional
            Current iteration count. Not used

        Returns
        -------
        proc_time : float
            Time spent in CRAM routines for all materials in [s]
        n_end : list of numpy.ndarray
            Concentrations at end of interval
        """
        # deplete to end using BOS rates
        proc_time, n_ce = self._timed_deplete(n_bos, rates, dt, _i)
        res_ce = self.operator(n_ce, source_rate)

        # deplete using two matrix exponentials
        list_rates = list(zip(rates, res_ce.rates))

        time_le1, n_inter = self._timed_deplete(
            n_bos, list_rates, dt, _i, matrix_func=celi_f1)

        time_le2, n_end = self._timed_deplete(
            n_inter, list_rates, dt, _i, matrix_func=celi_f2)

        return proc_time + time_le1 + time_le2, n_end


@add_params
class EPCRK4Integrator(Integrator):
    r"""Deplete using the EPC-RK4 algorithm.

    Implements an extended predictor-corrector algorithm with traditional
    Runge-Kutta 4 method. This algorithm is mathematically defined as:

    .. math::
        \begin{aligned}
        \mathbf{A}_1 &= h\mathbf{A}(\mathbf{n}_i) \\
        \hat{\mathbf{n}}_1 &= \exp \left ( \frac{\mathbf{A}_1}{2} \right ) \mathbf{n}_i \\
        \mathbf{A}_2 &= h\mathbf{A}(\hat{\mathbf{n}}_1) \\
        \hat{\mathbf{n}}_2 &= \exp \left ( \frac{\mathbf{A}_2}{2} \right ) \mathbf{n}_i \\
        \mathbf{A}_3 &= h \mathbf{A}(\hat{\mathbf{n}}_2) \\
        \hat{\mathbf{n}}_3 &= \exp \left ( \mathbf{A}_3 \right ) \mathbf{n}_i \\
        \mathbf{A}_4 &= h\mathbf{A}(\hat{\mathbf{n}}_3) \\
        \mathbf{n}_{i+1} &= \exp \left ( \frac{\mathbf{A}_1}{6} + \frac{\mathbf{A}_2}{3}
        + \frac{\mathbf{A}_3}{3} + \frac{\mathbf{A}_4}{6} \right ) \mathbf{n}_i.
        \end{aligned}
    """
    _num_stages = 4

    def __call__(self, n, rates, dt, source_rate, _i=None):
        """Perform the integration across one time step

        Parameters
        ----------
        n : list of numpy.ndarray
            List of atom number arrays for each material. Each array in the list
            contains the number of [atom] of each nuclide.
        rates : openmc.deplete.ReactionRates
            Reaction rates from operator
        dt : float
            Time in [s] for the entire depletion interval
        source_rate : float
            Power in [W] or source rate in [neutron/sec]
        _i : int, optional
            Current depletion step index, unused.

        Returns
        -------
        proc_time : float
            Time spent in CRAM routines for all materials in [s]
        n_end : list of numpy.ndarray
            Concentrations at end of interval
        """

        # Step 1: deplete with matrix A(y0) / 2
        time1, n1 = self._timed_deplete(n, rates, dt, _i, matrix_func=rk4_f1)
        res1 = self.operator(n1, source_rate)

        # Step 2: deplete with matrix A(y1) / 2
        time2, n2 = self._timed_deplete(n, res1.rates, dt, _i, matrix_func=rk4_f1)
        res2 = self.operator(n2, source_rate)

        # Step 3: deplete with matrix A(y2)
        time3, n3 = self._timed_deplete(n, res2.rates, dt, _i)
        res3 = self.operator(n3, source_rate)

        # Step 4: deplete with matrix built from weighted rates
        list_rates = list(zip(rates, res1.rates, res2.rates, res3.rates))
        time4, n4 = self._timed_deplete(n, list_rates, dt, _i, matrix_func=rk4_f4)

        return time1 + time2 + time3 + time4, n4


@add_params
class LEQIIntegrator(Integrator):
    r"""Deplete using the LE/QI CFQ4 algorithm.

    Implements the LE/QI Predictor-Corrector algorithm using the `fourth order
    commutator-free integrator <https://doi.org/10.1137/05063042>`_.

    "LE/QI" stands for linear extrapolation on predictor and quadratic
    interpolation on corrector. This algorithm is mathematically defined as:

    .. math::
        \begin{aligned}
        \mathbf{A}_{-1} &= \mathbf{A}(\mathbf{n}_{i-1}) \\
        \mathbf{A}_0 &= \mathbf{A}(\mathbf{n}_i) \\
        \mathbf{F}_1 &= \frac{-h_i}{12h_{i-1}} \mathbf{A}_{-1} + \frac{6h_{i-1}
            + h_i}{12h_{i-1}} \mathbf{A}_0 \\
        \mathbf{F}_2 &= \frac{-5h_i}{12h_{i-1}} \mathbf{A}_{-1} + \frac{6h_{i-1}
            + 5h_i}{12h_{i-1}} \mathbf{A}_0 \\
        \mathbf{n}_{i+1}^p &= \exp (h_i \mathbf{F}_1) \exp(h_i \mathbf{F}_2)
            \mathbf{n}_i \\
        \mathbf{A}_1 &= \mathbf{A}(\mathbf{n}_{i+1}^p) \\
        \mathbf{F}_3 &= \frac{-h_i^2}{12 h_{i-1} (h_{i-1} + h_i)} \mathbf{A}_{-1} +
              \frac{5 h_{i-1}^2 + 6 h_i h_{i-1} + h_i^2}{12 h_{i-1} (h_{i-1} +
              h_i)} \mathbf{A}_0 + \frac{h_{i-1}}{12 (h_{i-1} + h_i)} \mathbf{A}_1 \\
        \mathbf{F}_4 &= \frac{-h_i^2}{12 h_{i-1} (h_{i-1} + h_i)} \mathbf{A}_{-1} +
              \frac{h_{i-1}^2 + 2 h_i h_{i-1} + h_i^2}{12 h_{i-1} (h_{i-1} + h_i)}
              \mathbf{A}_0 + \frac{5 h_{i-1} + 4 h_i}{12 (h_{i-1} + h_i)} \mathbf{A}_1 \\
        \mathbf{n}_{i+1} &= \exp(h_i \mathbf{F}_4) \exp(h_i \mathbf{F}_3) \mathbf{n}_i
        \end{aligned}

    It is initialized using the CE/LI algorithm.
    """
    _num_stages = 2

    def __call__(self, n_bos, bos_rates, dt, source_rate, i):
        """Perform the integration across one time step

        Parameters
        ----------
        n_bos : list of numpy.ndarray
            List of atom number arrays for each material. Each array in the list
            contains the number of [atom] of each nuclide.
        bos_rates : openmc.deplete.ReactionRates
            Reaction rates from operator
        dt : float
            Time in [s] for the entire depletion interval
        source_rate : float
            Power in [W] or source rate in [neutron/sec]
        i : int
            Current depletion step index

        Returns
        -------
        proc_time : float
            Time spent in CRAM routines for all materials in [s]
        n_end : list of numpy.ndarray
            Concentrations at end of interval
        """
        if i == 0:
            if self._i_res < 1:  # need at least previous transport solution
                self._prev_rates = bos_rates
                return CELIIntegrator.__call__(
                    self, n_bos, bos_rates, dt, source_rate, i)
            prev_res = self.operator.prev_res[-2]
            prev_dt = self.timesteps[i] - prev_res.time[0]
            self._prev_rates = prev_res.rates
        else:
            prev_dt = self.timesteps[i - 1]

        # Remaining LE/QI
        bos_res = self.operator(n_bos, source_rate)

        le_inputs = list(zip(
            self._prev_rates, bos_res.rates, repeat(prev_dt), repeat(dt)))

        time1, n_inter = self._timed_deplete(
            n_bos, le_inputs, dt, i, matrix_func=leqi_f1)
        time2, n_eos0 = self._timed_deplete(
            n_inter, le_inputs, dt, i, matrix_func=leqi_f2)

        res_inter = self.operator(n_eos0, source_rate)

        qi_inputs = list(zip(
            self._prev_rates, bos_res.rates, res_inter.rates,
            repeat(prev_dt), repeat(dt)))

        time3, n_inter = self._timed_deplete(
            n_bos, qi_inputs, dt, i, matrix_func=leqi_f3)
        time4, n_eos1 = self._timed_deplete(
            n_inter, qi_inputs, dt, i, matrix_func=leqi_f4)

        # store updated rates
        self._prev_rates = copy.deepcopy(bos_res.rates)

        return time1 + time2 + time3 + time4, n_eos1


@add_params
class SICELIIntegrator(SIIntegrator):
    r"""Deplete using the SI-CE/LI CFQ4 algorithm.

    Implements the stochastic implicit CE/LI predictor-corrector algorithm
    using the `fourth order commutator-free integrator
    <https://doi.org/10.1137/05063042>`_.

    Detailed algorithm can be found in section 3.2 in `Colin Josey's thesis
    <https://dspace.mit.edu/handle/1721.1/113721>`_.
    """
    _num_stages = 2

    def __call__(self, n_bos, bos_rates, dt, source_rate, _i=None):
        """Perform the integration across one time step

        Parameters
        ----------
        n_bos : list of numpy.ndarray
            List of atom number arrays for each material. Each array in the list
            contains the number of [atom] of each nuclide.
        bos_rates : openmc.deplete.ReactionRates
            Reaction rates from operator
        dt : float
            Time in [s] for the entire depletion interval
        source_rate : float
            Power in [W] or source rate in [neutron/sec]
        _i : int, optional
            Current depletion step index. Not used

        Returns
        -------
        proc_time : float
            Time spent in CRAM routines for all materials in [s]
        n_end : list of numpy.ndarray
            Concentrations at end of interval
        op_result : openmc.deplete.OperatorResult
            Eigenvalue and reaction rates from intermediate transport
            simulations
        """
        proc_time, n_eos = self._timed_deplete(n_bos, bos_rates, dt, _i)
        n_inter = copy.deepcopy(n_eos)

        # Begin iteration
        for j in range(self.n_steps + 1):
            inter_res = self.operator(n_inter, source_rate)

            if j <= 1:
                res_bar = copy.deepcopy(inter_res)
            else:
                rates = 1/j * inter_res.rates + (1 - 1 / j) * res_bar.rates
                k = 1/j * inter_res.k + (1 - 1 / j) * res_bar.k
                res_bar = OperatorResult(k, rates)

            list_rates = list(zip(bos_rates, res_bar.rates))
            time1, n_inter = self._timed_deplete(
                n_bos, list_rates, dt, _i, matrix_func=celi_f1)
            time2, n_inter = self._timed_deplete(
                n_inter, list_rates, dt, _i, matrix_func=celi_f2)
            proc_time += time1 + time2

        # end iteration
        return proc_time, n_inter, res_bar


@add_params
class SILEQIIntegrator(SIIntegrator):
    r"""Deplete using the SI-LE/QI CFQ4 algorithm.

    Implements the Stochastic Implicit LE/QI Predictor-Corrector algorithm
    using the `fourth order commutator-free integrator
    <https://doi.org/10.1137/05063042>`_.

    Detailed algorithm can be found in Section 3.2 in `Colin Josey's thesis
    <https://dspace.mit.edu/handle/1721.1/113721>`_.
    """
    _num_stages = 2

    def __call__(self, n_bos, bos_rates, dt, source_rate, i):
        """Perform the integration across one time step

        Parameters
        ----------
        n_bos : list of numpy.ndarray
            List of atom number arrays for each material. Each array in the list
            contains the number of [atom] of each nuclide.
        bos_rates : list of openmc.deplete.ReactionRates
            Reaction rates from operator for all depletable materials
        dt : float
            Time in [s] for the entire depletion interval
        source_rate : float
            Power in [W] or source rate in [neutron/sec]
        i : int
            Current depletion step index

        Returns
        -------
        proc_time : float
            Time spent in CRAM routines for all materials in [s]
        n_end : list of numpy.ndarray
            Concentrations at end of interval
        op_result : openmc.deplete.OperatorResult
            Eigenvalue and reaction rates from intermediate transport
            simulation
        """
        if i == 0:
            if self._i_res < 1:
                self._prev_rates = bos_rates
                # Perform CELI for initial steps
                return SICELIIntegrator.__call__(
                    self, n_bos, bos_rates, dt, source_rate, i)
            prev_res = self.operator.prev_res[-2]
            prev_dt = self.timesteps[i] - prev_res.time[0]
            self._prev_rates = prev_res.rates
        else:
            prev_dt = self.timesteps[i - 1]

        # Perform remaining LE/QI
        inputs = list(zip(self._prev_rates, bos_rates,
                          repeat(prev_dt), repeat(dt)))
        proc_time, n_inter = self._timed_deplete(
            n_bos, inputs, dt, i, matrix_func=leqi_f1)
        time1, n_eos = self._timed_deplete(
            n_inter, inputs, dt, i, matrix_func=leqi_f2)

        proc_time += time1
        n_inter = copy.deepcopy(n_eos)

        for j in range(self.n_steps + 1):
            inter_res = self.operator(n_inter, source_rate)

            if j <= 1:
                res_bar = copy.deepcopy(inter_res)
            else:
                rates = 1 / j * inter_res.rates + (1 - 1 / j) * res_bar.rates
                k = 1 / j * inter_res.k + (1 - 1 / j) * res_bar.k
                res_bar = OperatorResult(k, rates)

            inputs = list(zip(self._prev_rates, bos_rates, res_bar.rates,
                              repeat(prev_dt), repeat(dt)))
            time1, n_inter = self._timed_deplete(
                n_bos, inputs, dt, i, matrix_func=leqi_f3)
            time2, n_inter = self._timed_deplete(
                n_inter, inputs, dt, i, matrix_func=leqi_f4)
            proc_time += time1 + time2

        # Store updated rates for next step
        self._prev_rates = copy.deepcopy(bos_rates)

        return proc_time, n_inter, res_bar


integrator_by_name = {
    'cecm': CECMIntegrator,
    'predictor': PredictorIntegrator,
    'adaptive': AdaptiveIntegrator,
    'cf4': CF4Integrator,
    'epc_rk4': EPCRK4Integrator,
    'si_celi': SICELIIntegrator,
    'si_leqi': SILEQIIntegrator,
    'celi': CELIIntegrator,
    'leqi': LEQIIntegrator
}
