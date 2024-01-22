# Dymas micromagnetic software.
# Author: Diego González Chávez
# email : diegogch@cbpf.br

import sympy
import numpy as np
from sympy.vector import dot
from . import LLG


class macroSpinSystem(object):
    def __init__(self,
                 energy_function,
                 coordinate_system,
                 m_variables,
                 u_amplitudes,
                 field_variable,
                 constants_dictionary,
                 m_inital_values,
                 alpha=0,
                 gamma=2.211e5,
                 units='SI',
                 ):
        self.nSpins = len(m_variables)
        self._m_symbols = sympy.Array(sympy.symbols('DymasMag:%d'
                                                    % (3*self.nSpins)))
        self._m_vars = [(self._m_symbols[i*3 + 0]*coordinate_system.i +
                         self._m_symbols[i*3 + 1]*coordinate_system.j +
                         self._m_symbols[i*3 + 2]*coordinate_system.k)
                        for i, _ in enumerate(m_variables)]
        _const_vars = [k for k in constants_dictionary.keys()]
        _const_values = [v for v in constants_dictionary.values()]
        self._energy_function = energy_function
        self._E = sympy.Subs(energy_function,
                             variables=m_variables + _const_vars,
                             point=self._m_vars + _const_values).doit()
        self._E_m = sympy.derive_by_array(self._E, self._m_symbols).doit()
        self._E_mm = sympy.derive_by_array(self._E_m, self._m_symbols).doit()

        self._H_symbol = field_variable
        self._H = 0*coordinate_system.k

        self._m_values = m_inital_values
        # Magnetic moment amplitudes
        self._u_amplitudes = sympy.Array(u_amplitudes)
        self.coordinate_system = coordinate_system

        self.alpha = np.ones((self.nSpins))*alpha
        self.gamma = np.ones((self.nSpins))*gamma

    def _to_npArray(self, expression):
        m_symbols_values = []
        for i, _m0 in enumerate(self._m_values):
            _n0 = _m0.normalize()
            m_symbols_values.append(dot(_n0, self.coordinate_system.i))
            m_symbols_values.append(dot(_n0, self.coordinate_system.j))
            m_symbols_values.append(dot(_n0, self.coordinate_system.k))
        sp_array = sympy.Subs(expression,
                              variables=self._m_symbols,
                              point=m_symbols_values).doit()
        # sp_array = sympy.Subs(sp_array,
        #                      variables=self._const_vars,
        #                      point=self._const_values).doit()
        sp_array = sympy.Subs(sp_array,
                              variables=[self._H_symbol],
                              point=[self.H]).doit()
        np_array = np.zeros(shape=sp_array.shape)
        it = np.nditer(np_array, flags=['multi_index'])
        for _ in it:
            np_array[it.multi_index] = sp_array[it.multi_index].evalf()
        return np_array

    @property
    def _Em(self):
        shape = (self.nSpins, 3)
        return self._to_npArray(self._E_m).reshape(shape)

    @property
    def _Emm(self):
        shape = (self.nSpins, 3, self.nSpins, 3)
        return self._to_npArray(self._E_mm).reshape(shape)

    @property
    def H_eff(self):
        '''
        Effective field array
        '''
        u_abs = self._to_npArray(self._u_amplitudes)
        return -self._Em / u_abs[:, None]

    @property
    def m(self):
        shape = (self.nSpins, 3)
        return self._to_npArray(self._m_symbols).reshape(shape)

    @m.setter
    def m(self, value):
        if isinstance(value, np.ndarray):
            self._m_values = []
            for i, _m0 in enumerate(value):
                self._m_values.append(_m0[0]*self.coordinate_system.i
                                      + _m0[1]*self.coordinate_system.j
                                      + _m0[2]*self.coordinate_system.k)
        if isinstance(value, list):
            self._m_values = value

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, value):
        self._H = value

    @property
    def N(self):
        shape = (self.nSpins, 3, self.nSpins, 3)
        u_abs = self._to_npArray(self._u_amplitudes)
        m_eq_h_eq = np.einsum('i,ia,ia->i', 1/u_abs, self._Em, self.m)
        return (-self._Emm / u_abs[:, None, None, None]
                + np.kron(np.diag(m_eq_h_eq), np.eye(3)).reshape(shape))

    def Minimize(self,
                 max_steps=1000,
                 start_max_angle=0.05,
                 target_max_angle=1E-3):

        max_changes = np.ones(10)
        max_changes[-1] = 0
        max_angle = start_max_angle

        work_m = self.m.copy()
        shape = (self.nSpins, 3)

        u_abs = self._to_npArray(self._u_amplitudes)
        _Em = sympy.Subs(self._E_m,
                         variables=[self._H_symbol],
                         point=[self.H]).doit()
        _Em = sympy.lambdify(self._m_symbols, _Em)

        for i in range(max_steps):
            H_eff = -_Em(*work_m.flatten()).reshape(shape) / u_abs[:, None]
            D = np.einsum('xab,xb->xa',
                          LLG.damping_operator_LL(self.alpha,
                                                  self.gamma,
                                                  work_m),
                          H_eff)  # [rad?/s]

            if (max_changes.std() / max_changes.ptp()) > 0.4:
                max_angle *= 0.90

            maxChange = np.abs(D).max()  # in rads/s
            if maxChange == 0:
                break

            work_m = work_m + D*max_angle/maxChange
            work_m = work_m/np.linalg.norm(work_m, axis=-1, keepdims=True)

            max_changes = np.take(max_changes, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            max_changes[-1] = maxChange

            if max_angle <= target_max_angle:
                break
        self.m = work_m.copy()