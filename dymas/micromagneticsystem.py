# Dymas micromagnetic software.
# Author: Diego González Chávez
# email : diegogch@cbpf.br

import numpy as np
import numpy.typing as npt
import LLG
import demag
import exchange

def _not_implemented_method(*args, **kargs):
    print('Not implemented!')


class Models(object):
    pass


class System(object):
    __slots__ = ['mesh_n', 'mesh_cell',
                 'm', 'Ms', 'mask',
                 'Dynamics', 'Energies',
                 '_minimize_method',
                 '_evolve_method',
                 '_eigen_method',
                 '_Chi_method',
                 '_states',
                 '__dict__']

    def __init__(self) -> None:
        self._minimize_method = _not_implemented_method
        self._evolve_method = _not_implemented_method
        self._eigen_method = _not_implemented_method
        self._Chi_method = _not_implemented_method
        self._states = []

    def save(self, file_name):
        np.savez_compressed(file_name,
                            mesh_cell=self.mesh_cell,
                            mesh_n=self.mesh_n,
                            Ms=self.Ms,
                            alpha=self.alpha,
                            gamma=self.gamma,
                            K_total=self.K_Total,
                            H=self.H,
                            m=self.m)

    @classmethod
    def fromFile(cls, file_name: str):
        instance = cls()
        data = np.load(file_name + 'npz')
        instance.mesh_cell = data['mesh_cell']
        instance.mesh_n = data['mesh_n']
        instance.Ms = data['Ms']
        instance.alpha = data['alpha']
        instance.gamma = data['gamma']
        instance.K_Total = data['K_Total']
        instance.H = data['H']
        instance.m = data['m']

    @classmethod
    def fromUbermagSystem(cls, ub_system):
        mesh_n = ub_system.m.mesh.n
        mesh_cell = ub_system.m.mesh.cell

        instance = cls()
        instance.mesh_n = mesh_n
        instance.mesh_cell = mesh_cell
        instance.Ms = np.atleast_1d(ub_system.m.norm.array)[..., 0]
        mask = (instance.Ms == 0)
        instance.m = np.empty_like(ub_system.m.array)
        instance.m[mask] = [0, 0, 0]
        instance.m[~mask] = np.asarray(ub_system.m.array)[~mask] \
            / instance.Ms[~mask][..., None]
        instance.mask = mask
        instance.Energies, instance.Dynamics = parse_ubermag_system(ub_system)


        # get kernels
        K_demag = demag.Demag_Kernel(*mesh_n, *mesh_cell)
        A_eff = -2*instance.Energies['BasicExchange']['A']
        A_eff[mask] = 0
        A_eff[~mask] = A_eff[~mask]/instance.Ms[~mask]
        K_exchange = exchange.Exchange_Kernel(*mesh_n, *mesh_cell, A_eff)
        instance.H = instance.Energies['StaticZeeman']['H']
        instance.alpha = instance.Dynamics['LLG_Damping']['alpha']
        instance.gamma = instance.Dynamics['LLG_Precession']['gamma']
        instance.K_Total = np.einsum('ijkalmnb,lmn->ijkalmnb',
                                     K_demag + K_exchange,
                                     instance.Ms)

        #use copies in order to free the original array memory.
        instance.m = instance.m[~mask].copy()
        instance.H = instance.H[~mask].copy()
        instance.K_Total = instance.K_Total[~mask, ...][:, :, ~mask, :].copy()
        instance.alpha = instance.alpha[~mask].copy()
        instance.gamma = instance.gamma[~mask].copy()
        return instance


def parse_ubermag_system(ub_system):
    def get_names_and_attributes(terms):
        out = []
        for t in terms:
            item = {'class': str(t.__class__)}
            for a in t._allowed_attributes:
                try:
                    value = t.__getattribute__(a)
                except AttributeError:
                    next
                if 'ubermagutil.typesystem.descriptors.Parameter' \
                        not in str(type(value)):
                    item[a] = value
            out.append(item)
        return out

    def value_to_array(value, mask=None):
        # convert fields, scalars, vectors and region_dicts to field_arrays
        # TODO use masked arrays
        if 'discretisedfield.field.Field' in str(type(value)):
            return value.array
        elif np.isscalar(value):
            # ijk shape
            return np.full(ub_system.m.array.shape[:-1], value)
        elif np.shape(value) == (3,):
            # ijka shape
            return np.full_like(ub_system.m.array, np.atleast_1d(value))
        elif isinstance(value, dict):
            # ijk shape
            # we have a dict with a (scalar?) value for each subregion
            out = np.zeros(ub_system.m.array.shape[:-1])

            # Remove interfaces (unimplemented)
            for region_key in value.keys():
                if ':' in region_key:
                    value.pop(region_key)

            for xyz in ub_system.m.mesh:
                for region, val in value.items():
                    if ub_system.m.mesh.subregions[region].__contains__(xyz):
                        out[ub_system.m.mesh.point2index(xyz)] = val
            return out

    mask = (ub_system.m.norm.array[:, :, :, 0] == 0)

    dynamics = {}
    for data in get_names_and_attributes(ub_system.dynamics):
        if 'micromagneticmodel.dynamics.precession.Precession' in data['class']:
            dynamics['LLG_Precession'] = \
                {'gamma': value_to_array(data['gamma0'], mask)}
        elif 'micromagneticmodel.dynamics.damping.Damping' in data['class']:
            dynamics['LLG_Damping'] =\
                {'alpha': value_to_array(data['alpha'], mask)}

    energies = {}
    for data in get_names_and_attributes(ub_system.energy):
        if 'micromagneticmodel.energy.exchange.Exchange' in data['class']:
            energies['BasicExchange'] = {'A': value_to_array(data['A'], mask)}

        elif 'micromagneticmodel.energy.demag.Demag' in data['class']:
            energies['BasicDemag'] = {}

        elif 'micromagneticmodel.energy.zeeman.Zeeman' in data['class']:
            if 'func' not in data.keys():
                energies['StaticZeeman'] = {'H': value_to_array(data['H'])}
            else:
                energies['uninplemented_Zeeman'] = {}
    return energies, dynamics

def Evolve(system: System,
           t: float,
           max_steps:int=5000,
           max_angle:float=0.5E-3,
           ):
    
    #Flat versions of the arrays
    mSize = system.m.size
    K = system.K_Total.reshape(mSize, mSize)
    H_DC = system.H.ravel()
    
    t_evolved = 0
    for i in range(max_steps):
        H_eff = H_DC + K @ system.m.ravel()
        H_eff.shape = system.m.shape
        dT = np.einsum('xab,xb->xa',
                      LLG.damping_operator_LL(system.alpha,
                                              system.gamma,
                                              system.m)
                      + LLG.torque_operator_LLG(system.alpha,
                                                system.gamma,
                                                system.m),
                      H_eff)  # [rad?/s]

        maxChange = np.abs(dT).max()  # in rads/s
        if maxChange == 0:
            break
        step = min(max_angle/maxChange, t-t_evolved)
        t_evolved += step

        system.m = system.m + dT*step
        system.m = system.m/np.linalg.norm(system.m, axis=-1, keepdims=True)
        if t_evolved >= t:
            break


def Minimize(system: System,
              max_steps=1000,
              start_max_angle=0.05,
              target_max_angle=2E-5):

    max_changes = np.ones(10)
    max_changes[-1] = 0
    max_angle = start_max_angle

    #Flat versions of the arrays
    mSize = system.m.size
    K = system.K_Total.reshape(mSize, mSize)
    H_DC = system.H.ravel()

    for i in range(max_steps):
        H_eff = H_DC + K @ system.m.ravel()
        H_eff.shape = system.m.shape
        D = np.einsum('xab,xb->xa',
                      LLG.damping_operator_LL(system.alpha,
                                              system.gamma,
                                              system.m),
                      H_eff)  # [rad?/s]

        if (max_changes.std() / max_changes.ptp()) > 0.4:
            max_angle *= 0.90

        maxChange = np.abs(D).max()  # in rads/s
        if maxChange == 0:
            break

        system.m = system.m + D*max_angle/maxChange
        system.m = system.m/np.linalg.norm(system.m, axis=-1, keepdims=True)

        max_changes = np.take(max_changes, [1,2,3,4,5,6,7,8,9,0])
        max_changes[-1] = maxChange

        if max_angle <= target_max_angle:
            break