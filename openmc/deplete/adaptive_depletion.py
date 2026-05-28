import openmc.deplete
import openmc.mgxs


import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from time import time
import types
import h5py
from functools import reduce

# thermal groups
magnox = openmc.mgxs.GROUP_STRUCTURES['MPACT-69']
shem = openmc.mgxs.GROUP_STRUCTURES['SHEM-361']
casmo = openmc.mgxs.GROUP_STRUCTURES['CASMO-70']

# fast groups
ecco = openmc.mgxs.GROUP_STRUCTURES['ECCO-33']
vitaminj = openmc.mgxs.GROUP_STRUCTURES['VITAMIN-J-175']


energy_groups = reduce(np.union1d, (magnox, shem, casmo, ecco, vitaminj))
print(len(energy_groups))


'''
the PredictorIntegrator has a tendency to change results interpretation if it
gets an array of one component only, which is what we will do, we will 
integrate one step at a time, therefore the patch_integrator_instance function
is created to tell it to not make such assumptions
'''

def _patch_integrator_instance(integrator):
    orig = integrator._get_bos_data_from_restart

    def patched(self, source_rate, n):
        n = list(self.operator.prev_res[-1].data)
        return self._get_bos_data_from_operator(0, source_rate, n)

    integrator._get_bos_data_from_restart = types.MethodType(patched,
                                                             integrator)
    return orig


def _get_atoms_count_at_step_i(step: int,
                               depletion_file: str,
                               material_id: str | int,
                               nuclides_list: list) -> dict:

    if isinstance(material_id, int):

        material_id = str(material_id)


    with h5py.File(depletion_file, "r") as h5:
        number = h5["/number"]

        if material_id not in h5["/materials"]:

            available = list(h5["/materials"].keys())
            raise KeyError(f"mat_id={material_id} not found. "
                           f"Available material IDs include: {available}")


        material_idx = int(h5["/materials"][material_id].attrs["index"])

        nuc_to_idx = {}
        missing = []

        for nuc in nuclides_list:
            if nuc in h5["/nuclides"]:

                nuc_to_idx[nuc] = int(h5["/nuclides"][nuc].attrs["atom number "
                                                                 "index"])

            else:
                missing.append(nuc)
                raise KeyError(f'the following nuclides are not found in the'
                               f' depletion results file: {missing}')

        row = number[step, material_idx, :]

        atoms_count = {nuc: float(row[idx]) for nuc, idx in nuc_to_idx.items()}

        atoms_count = dict(sorted(atoms_count.items()))

        return atoms_count


REACTIONS_PER_NUCLIDE = 7 # in the microxs file, there are 7 tallied reactions
                          # of which xs are given, (n,gamma); (n,2n); (n,p);
                          # (n,a); (n,3n); (n,4n); (n, fission). we will
                          # probably not need this variable, but it is defined
                          # just in case we do


class AdaptiveDepletion:

    def __init__(self, chain_file,
                 flux_file= None,
                 microxs_file= None,
                 inventory_deviation_tolerance = 1,
                 timesteps = None,
                 power = None,
                 model: openmc.Model= None,
                 material_to_deplete: openmc.Material= None,
                 energies= None,
                 depletion_file= None):

        self.flux_file = flux_file
        self.microxs_file = microxs_file
        self.inventory_deviation_tolerance = inventory_deviation_tolerance
        #self.delta_m = self.mass_deviation_tolerance
        self.chain_file = chain_file
        self.timesteps = timesteps
        self.power = power
        self.model = model
        self.material_to_deplete = material_to_deplete
        self.energies = energies
        self.depletion_file = depletion_file
        self.transport_trigger_step = None
        self.delta_m = None
        self.RR_N = None
        self.w =None

    def _get_reaction_rates_per_atom(self, return_type: str = "dict"):

        # Your reaction multipliers
        REACTION_WEIGHTS = {
            "(n,gamma)": 1.0,
            "(n,p)": 1.0,
            "(n,a)": 1.0,
            "(n,2n)": 3.0,
            "(n,3n)": 4.0,
            "(n,4n)": 5.0,
            "fission": 6.46,
        }

        # Load flux and normalize to a shape (same as you already do)
        initial_flux = np.load(self.flux_file)[0]
        normalized_flux = initial_flux / initial_flux.sum()

        # Read all columns needed for reaction-specific weighting
        df = pd.read_csv(self.microxs_file, usecols=["nuclides", "reactions", "groups", "xs"]).copy()

        # Clean nuclide & reaction strings
        df["nuclides"] = df["nuclides"].astype(str).str.strip()
        df["reactions"] = df["reactions"].astype(str).str.strip()

        # Ensure xs is numeric
        df["xs"] = pd.to_numeric(df["xs"], errors="coerce").fillna(0.0)

        # Map reaction -> weight (error out if something unexpected appears)
        df["w_rxn"] = df["reactions"].map(REACTION_WEIGHTS)
        if df["w_rxn"].isna().any():
            missing = sorted(df.loc[df["w_rxn"].isna(), "reactions"].unique())
            raise KeyError(f"Missing reaction weights for reactions: {missing}")

        # Use the explicit group index to grab the matching flux value (groups are 1-indexed)
        g = df["groups"].to_numpy()
        if g.min() < 1 or g.max() > len(normalized_flux):
            raise ValueError(
                f"Energy group index out of bounds: min={g.min()}, max={g.max()}, "
                f"but flux length is {len(normalized_flux)}"
            )

        # RR/N per row, weighted by reaction multiplier, before summing
        df["RR/N"] = df["xs"] * normalized_flux[g - 1] * df["w_rxn"]  # arbitrary unnormalized units

        # Sum over all energy groups and all reactions for each nuclide
        self.RR_N = (
            df.groupby("nuclides", as_index=False)["RR/N"]
            .sum()
            .rename(columns={"RR/N": "RR/N"})
        )

        if return_type == "df":
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", None)
            return self.RR_N

        elif return_type == "dict":
            unsorted_dict = dict(zip(self.RR_N["nuclides"], self.RR_N["RR/N"]))
            self.RR_N = dict(sorted(unsorted_dict.items()))
            return self.RR_N

        else:
            raise TypeError("return type must be either df or dict")


    def _get_weighing_factors(self, alpha):

        self.RR_N = self._get_reaction_rates_per_atom()

        self.weighting_factors = {}

        for nuc in self.RR_N.keys():

            self.weighting_factors[nuc] = self.RR_N[nuc]**alpha

        return self.weighting_factors


    '''
    after getting the weighing factors, the best and most efficient option is to take in the names of 
    nuclides of which weighing factors are calculated and exist in the microxs.csv file, gather those in 
    a list like the one previously called selected_nuclides, then perform the the first loop which is
    depletion timesteps, and the second loop inside which goes and looks for mass of selected nuclide 
    change at that timestep and compare it to t0 then sums and prints the total deviation for that timestep
    '''
# _get_weighing_factors is the actual method used during an algorithm run
# the following function is a standalone version of calculating deviations
# constructed before implementation
    def get_deviations(self, w, material_to_deplete, include_unweighted= False):

        self.material_to_deplete = material_to_deplete

        self.w = w

        #delta_m = 1
        # load results from the depletion file
        results = openmc.deplete.Results(filename= self.depletion_file)

        # get number of timesteps to compute deviation for each timestep later on
        times = results.get_times(time_units= 'd')

        # get the nuclides with existing reaction rates data in a list as strings to call each by name later on
        if not isinstance(self.w, dict):

            self.w = self.w.set_index('nuclides')['w'].to_dict()

        # load nuclide names which have weighting factors
        monitored_nuclides = list(self.w.keys())

        weighted_deviations = {}
        unweighted_deviations = {}

        for i in range(len(times)):

            step_i = results[i]

            # load nuclide names which exist in the depletion_results.h5 file
            depleted_nuclides = list(step_i.index_nuc.keys())

            # start counting weighted deviation for that step (gets reassigned to 0 at the start of each new iteration)
            step_weighted_deviation = 0

            # unweighted deviation is the exact same as weighted deviations except that weighting factors are not taken into account
            step_unweighted_deviation = 0

            # verify nuclide that has weighting factor exits in the depletion_results.h5 file
            for nuc in monitored_nuclides:
                if nuc in depleted_nuclides:

                    # calculate mass deviation for step i by comparing it to step 0 (the value is absolute here of course)
                    weighed_mass = abs((results.get_mass(mat=self.material_to_deplete, nuc=nuc, mass_units='g')[1][i]) - (results.get_mass(mat=self.material_to_deplete, nuc=nuc, mass_units='g')[1][0])) * self.w[nuc]
                    step_weighted_deviation += weighed_mass

            if include_unweighted:
                for nuc in depleted_nuclides:
                    unweighted_mass = abs((results.get_mass(mat=self.material_to_deplete, nuc=nuc, mass_units='g')[1][i]) - (results.get_mass(mat=self.material_to_deplete, nuc=nuc, mass_units='g')[1][0]))
                    step_unweighted_deviation += unweighted_mass

            # save result of that step into the dictionary next to the step number
            weighted_deviations[i+1] = step_weighted_deviation
            unweighted_deviations[i+1] = step_unweighted_deviation

            # when the deviation of that step exceeds the tolerance, make sure to call out the step and print out a statement
            if weighted_deviations[i+1] >= self.inventory_deviation_tolerance and self.transport_trigger_step is None:
                self.transport_trigger_step = i
                print(f'step {i + 1}: WEIGHTED DEVIATION = {weighted_deviations[i + 1]:.3f},   UNWEIGHTED DEVIATION = {unweighted_deviations[i + 1]:.3f}\n\n!TRIGGER TRANSPORT! {self.transport_trigger_step + 1}\n')

            else:
                print(f'step {i+1}: WEIGHTED DEVIATION = {weighted_deviations[i+1]:.3f},   UNWEIGHTED DEVIATION = {unweighted_deviations[i+1]:.3f}')

        # estimate the number of transports left by dividing the maximum deviation which occurs in the last step by the mass tolerance
        self.transports_left = round(weighted_deviations[len(times)] / self.inventory_deviation_tolerance)



        print(f'estimated number of transports left: {self.transports_left}')

        # the return here is really not that important, what really matters is the printing statements and information being extracted
        # a method similar to this may be the bridge which helps the user decide how their system is behaving and how they want to
        # conduct depletion
        return self.transport_trigger_step

    def deplete(self, material_to_deplete,
                 model,
                 timesteps,
                 power,
                 timestep_units,
                 energies,
                 alpha):

        '''
        this method will basically use a combination of IndependentOperator and
        PredictorIntegrator to do transport-adaptive depletion. the transport-
        adaptive approach is an especially useful approach for depletion problems
        were there are numerous time steps. It simply works by automatically
        triggering transport only at time steps were it is predicted that the
        flux spectrum has changed and there is a need to update our frozen
        flux input. users can either choose a tolerance limit based on desired
        fidelity or choose the affordable number of transports they want to do
        for their depletion problem.
        '''

        '''
        the main variable we need to solve for per integration iteration is the
        neutronically weighted percent deviation, we will basically compute
        number of atoms per nuclide multiplied by the reaction rates per nuclide
        (RR/N) for that nuclide and sum that for all nuclides in the materials,
        let's call this variable 'N', A will be computed at t(0) and compared to
        A at t(i), where t(0) is the interval at the latest transport, then our
        percent deviation simply becomes '(N(i) - N(0)) / N(0)', we will monitor
        and compute this quantity after each depletion interval, and whenever it
        exceeds a certain tolerance limit, we will trigger a transport
        '''

        if len(power) == len(timesteps):
            intervals_len = len(timesteps)
        else:
            err_msg = 'power and timesteps are not of the same length'
            raise ValueError(err_msg)

        percent_deviations_0 = []

        t0 = time()


        self.material_to_deplete = material_to_deplete
        # self.material_to_deplete.id = 1

        self.transports_counter = 0 # bookkeeping to keep track of number of transports triggered
        real_step = 0 # bookkeeping to keep track of real step number when we get in and out the for loop
        t0_step = 0

        # set previous results to None at first step, but then to the actual depletion_results.h5 file
        prev = None
        final_step = False

        #flux, sigma = openmc.deplete.get_microxs_and_flux(model=model, energies=energies, domains=[self.material_to_deplete], chain_file=self.chain_file)


        # we will fill this list later with the steps in which transport was triggered
        transport_trigger_steps = []

        #monitored_nuclides = list(self.w.keys())

        # the dictionaries below are there to collect atoms names and corresponding counts, atoms_count_0_dict will be collected and updated at transports only
        # while atoms_count_i_dict will be collected and updated at each depletion interval, numpy arrays will be extracted from them to do operations faster

        atoms_count_0_dict = {}
        atoms_count_i_dict = {}
        deviations_collection = {}



        weighted_atoms_difference = []

        weighted_deviations = {}

        # we will set it off to tolerance to kick-start the while loop and reset it to zero at the start of every while loop iteration
        step_percent_deviation = self.inventory_deviation_tolerance
        # t0 here is updated, and it corresponds to the step at which the latest transport took place, it will also be updated after every transport


        # start the while loop, which will only be broken until two conditions are met (1) we have reached the last step
        # (2) the current deviation within than our tolerance, each iteration of this while loop is equivalent of exactly one transport
        while (step_percent_deviation >= self.inventory_deviation_tolerance
               and real_step <= intervals_len):

            # set deviation to zero at the beginning of each transport step
            step_percent_deviation = 0
            weighted_atoms_count_0 = []


            t1 = time()

            if real_step == 0:
                """
                flux = np.load("flux_1.npy")
                sigma = [openmc.deplete.MicroXS.from_csv("microxs_1.csv")]
                """
                flux, sigma = openmc.deplete.get_microxs_and_flux(model=model,
                                                                  energies=energies,
                                                                  domains=[self.material_to_deplete],
                                                                  chain_file=self.chain_file)

            else:
                #print(model.materials)
                flux, sigma = openmc.deplete.get_microxs_and_flux(model=model,
                                                                  energies=energies,
                                                                  domains=[self.material_to_deplete],
                                                                  chain_file=self.chain_file,
                                                                  material_id= self.material_to_deplete.id,
                                                                  material_densities=atoms_count_i_dict)

            self.transports_counter += 1
            np.save(f"flux_{self.transports_counter}.npy", flux)
            microxs = sigma[0]
            '''
            print(f'microxs.data: {microxs.data}')
            print(f'microxs.nuclides: {microxs.nuclides}')
            print(f'microxs.reactions: {microxs.reactions}')
            '''
            microxs.to_csv(f"microxs_{self.transports_counter}.csv")

            self.microxs_file = f"microxs_{self.transports_counter}.csv"
            self.flux_file = f"flux_{self.transports_counter}.npy"

            t2 = time()

            print(f'time spent in transport is {(t2-t1) / 60} minutes')

            # obtain/update reaction rates per atom and weighting factors for that new flux
            nuclides_weighting_factors = self._get_weighing_factors(alpha= alpha)
            #print(nuclides_weighting_factors.keys())
            # nuclides_weighting_factors = np.array(list(nuclides_weighting_factors.values()), dtype=np.float64)

            """
            # plot normalized flux spectrum to observe and monitor potential spectrum changes
            plt.figure(figsize=(6, 4), dpi=1000)
            diff_flux = flux[0] / np.diff(energy_groups)
            plt.plot(energy_groups[:-1], diff_flux / np.sum(diff_flux))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(1.0e1, 10e06)
            plt.ylim(1.0e-10, 0.5)
            plt.grid(True)
            plt.title(f'flux shape at {self.transports_counter} transports')
            plt.show()
            """
            # start the for loop which will iterate over time and power steps, and integrate for each step individually using the
            # IndependentOperator and PredictorIntegrator and evaluate nuclides evolution at each step
            for i, (dt, p) in enumerate(zip(timesteps, power)):

                real_step += 1
                weighted_atoms_count_i = []

                op = openmc.deplete.IndependentOperator(
                    materials=[self.material_to_deplete],
                    fluxes=flux,
                    micros=sigma,
                    chain_file=self.chain_file,
                    normalization_mode='fission-q',
                    prev_results=prev,
                )

                integrator = openmc.deplete.PredictorIntegrator(
                    op,
                    timesteps=[dt],
                    timestep_units=timestep_units,
                    power=[p]
                )

                #orig = _patch_integrator_instance(integrator)
                integrator.integrate(path='depletion_results.h5')
                #integrator._get_bos_data_from_restart = orig

                results = openmc.deplete.Results(filename='depletion_results.h5')

                # set previous results so the chain is never broken
                prev = results


                step_i = results[real_step]
                depleted_nuclides = list(step_i.index_nuc.keys())

                step_weighted_differences = []

                # iterate over all nuclides
                for nuc in nuclides_weighting_factors.keys():
                    if nuc in depleted_nuclides:

                        if i == 0:

                            atoms_count_0_dict[nuc] = results.get_atoms(
                                mat=self.material_to_deplete,
                                nuc=nuc,
                                nuc_units='atoms')[1][t0_step]

                            weighted_atoms_count_0.append(atoms_count_0_dict[nuc] * nuclides_weighting_factors[nuc])


                        atoms_count_i_dict[nuc] = results.get_atoms(
                            mat=self.material_to_deplete,
                            nuc=nuc,
                            nuc_units='atoms')[1][-1]


                        weighted_atoms_count_i.append(atoms_count_i_dict[nuc] * nuclides_weighting_factors[nuc])

                    #print(f'atomic count dictionary: {atoms_count_i_dict}')
                    #print(f'weighted atomic count dictionary: {weighted_atoms_count_i}')
                if i == 0:
                    weighted_atoms_count_0 = np.array(weighted_atoms_count_0)

                weighted_atoms_count_i = np.array(weighted_atoms_count_i)

                step_weighted_differences.append(abs(
                    weighted_atoms_count_i - weighted_atoms_count_0))

                step_percent_deviation = (np.sum(step_weighted_differences) / np.sum(weighted_atoms_count_0)) * 100
                deviations_collection[real_step] = step_percent_deviation

                '''
                print(f'atoms_count_0_dict: {atoms_count_0_dict}')
                print(f'atoms_count_i_dict {atoms_count_i_dict}')
                print(f'nuclides_weighting_factors: {nuclides_weighting_factors}')
                print(f'weighted_atoms_count_0: {weighted_atoms_count_0}')
                print(f'weighted_atoms_count_i: {weighted_atoms_count_i}')
                print(f'step_percent_deviation {step_percent_deviation}')
                '''

                '''
                weighed_atom_deviation =  (abs((atoms_count_i - atoms_counts_0)) * nuclides_weighting_factors[nuc]) / (atoms_counts_0 * nuclides_weighting_factors[nuc])

                # dump the weighted atoms_count of nuclides into the step_weighted_deviation counter
                step_percent_deviation += weighed_atom_deviation


                weighted_deviations[real_step] = step_weighted_deviation
                step_percent_deviation = weighted_deviations[real_step]
                
                '''
                # I'm pretty sure this and the next if conditions are the same but at this point I am too afraid to touch either (:
                if step_percent_deviation >= self.inventory_deviation_tolerance:
                    self.transport_trigger_step = real_step
                    print(
                    f'step {real_step}: PERCENT DEVIATION = {step_percent_deviation:.3f}\n\nTRIGGERING TRANSPORT at step {real_step} with p= {p:.3f} and dt= {dt:.3f}...\n')

                else:
                    print(f'\nstep {real_step}: PERCENT DEVIATION = {step_percent_deviation:.3f} %\n')

                # when our tracker (step_percent_deviation) records that our mass exceeded our tolerance, we need to update our inputs and break the for loop
                if step_percent_deviation >= self.inventory_deviation_tolerance:

                    # update materials and model to prepare for the next transport
                    """
                    updated_materials = results.export_to_materials(-1)
                    #print(updated_materials[0])

                    model.materials = updated_materials

                    for new_mat in updated_materials:
                        if new_mat.name == self.material_to_deplete.name:
                            self.material_to_deplete = new_mat
                        for old_mat in model.materials:
                            if new_mat.name == old_mat.name:
                                new_mat.id = old_mat.id
                                break

                    #print(model.materials[0])
                    """
                    # time and power steps need to exclude the slices which were already integrated
                    timesteps = timesteps[i+1:]
                    power = power[i+1:]

                    '''
                    if len(power) == len(timesteps):
                        if len(power) == 1:
                            final_step = True
                    else:
                        err_msg = 'power and timesteps are not of the same length'
                        raise ValueError(err_msg)
                    '''

                    # record steps in which transport was triggered
                    transport_trigger_steps.append(real_step)

                    # record the new t0 that acts as a reference to measure deviation after each transport
                    t0_step = real_step

                    # break the for loop and go back to the while loop
                    break

        '''
        with h5py.File("depletion_results.h5", "a") as f:
            f.create_dataset("adaptive/transport_trigger_steps", data=np.array(transport_trigger_steps, dtype=int))
        '''
        tf = time()
        print(f'\ndeviations: {deviations_collection}')
        print(f'\nnumber of triggers: {len(transport_trigger_steps)}')
        print(f'\ntrigger steps: {transport_trigger_steps}')
        print(f'\ntotal time passed: {(tf-t0) / 3600} hours')