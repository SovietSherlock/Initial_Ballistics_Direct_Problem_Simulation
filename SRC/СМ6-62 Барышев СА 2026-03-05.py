from math import *
from typing import Any
from numpy import dtype, float64, ndarray
from ODE_solvers import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyballistics import ozvb_lagrange



class Cannon_System_Parameters:
    """Класс параметров орудийной системы"""

    def __init__(self):
        """Перезапись параметров из файла с исходными данными для далнейшего анализа:"""
        ds = pd.read_csv('СМ6-62 Барышев СА 2026-03-05 data.csv', skiprows=[10]).set_index('var').to_dict()
        # чтение файла "СМ6-62 Барышев СА 2026-03-05 data.csv":
        # 1. Открытие csv файла с начальными данными, игнорирование 11 строки (строка с обозначением пороха),
        # 2. Преобразование индексов строк в обозначение параметров таблицы
        # 3. Преобразование таблицы в словарь ds = {'value': {'параметр_1': значение_1, 'параметр_2': значение_2, ...}}
        vals = ds['value']
        # извлечение значений по ключу 'value' в отдельный словарь vals = {'параметр_1': значение_1, 'параметр_2': значение_2, ...}

        "Параметры орудийного ствола:"
        self.d = vals['d'] # калибр орудия, м
        self.q = vals['q'] # масса пороха, кг
        self.n_s = vals['n_s'] # коэффициент увеличения площади поперечного сечения канала ствола за счет нарезов
        self.l_m_d = vals['l_m/d'] # относительная длина ведущей части канала ствола орудия, м

        "Условия заряжания:"
        self.K = vals['K'] # поправочный коэффициент, учитывающий второстепенные работы газа
        self.p_ign = vals['p_ign'] # начальное давление воспламенения, Па
        self.p_0 = vals['p_0'] # давление форсирования, Па
        self.Delta = vals['Delta'] # плотность заряжания, кг/м^3
        self.omega_q = vals['omega/q'] # относительная масса заряда

        "Параметры пороха ДГ-3 20/1:"
        self.I_en = vals['I_en'] # импульс пороха, Па*с
        self.f_n = vals['f_n'] # сила пороха, Дж/кг
        self.k = vals['k'] # показатель адиабаты
        self.T_v = vals['T_v'] # температура горения пороха, К
        self.delta = vals['delta'] # плотность пороха, кг/м^3
        self.b = vals['b'] # коволюм единицы массы газа, м^3/кг
        self.z_e = vals['z_e'] # толщина сгоревшего свода порохового элемента в конце горения
        self.kappa_1 = vals['kappa_1'] # коэффициент формы порохового зерна
        self.kappa_2 = vals['kappa_2'] # коэффициент формы порохового зерна
        self.lambda_1 = vals['lambda_1'] # коэффициент формы порохового зерна
        self.lambda_2 = vals['lambda_2'] # коэффициент формы порохового зерна
        self.k_I = vals['k_I'] # коэффициент температурной чувствительности полного импульса давления пороховых газов к начальной температуре снаряда, К^-1
        self.k_f = vals['k_f'] # коэффициент температурной чувствительности силы пороха к начальной температуре снаряда, К^-1


        "Табличные параметры нормальных условий:"
        self.T_ref = vals['T_ref'] # температура при нормальных условиях, К
        self.Pr = vals['Pr'] # число Прандля
        self.lambda_g = vals['lambda_g'] # коэффициент теплопроводности пороховых газов, Вт/(м*К)

        "Параметры динамической вязкости пороховых газов (константы Сазерленда):"
        self.mu_g0 = vals['mu_g0'] # референсная вязкость, Па*с
        self.T_0s = vals['T_0s'] # референсная начальная температура, К
        self.T_cs = vals['T_cs'] # референсная конечная температура, К

        "Характеристики материала ствола:"
        self.rho_b = vals['rho_b'] # плотность стали, из которой изготовлен ствол, кг/м^3
        self.c_b = vals['c_b'] # удельная теплоемкость стали, из которой изготовлен ствол, Дж/(кг*К)
        self.lambda_b = vals['lambda_b'] # коэффициент теплопроводности стали, из которой изготовлен ствол, Вт/(м*К)

        "Параметры расчета давления воздушного столба перед снарядом (параметры атмосферы):"
        self.p_a0 = vals['p_a0'] # атмосферное давление, Па
        self.k_a = vals['k_a'] # показатель адиабаты воздуха
        self.c_a0 = vals['c_a0'] # скорость звука в атмосфере, м/с

        "Начальная температура заряда и стенок ствола орудия:"
        self.T_01 = vals['T_01'] # -50 град С
        self.T_02 = vals['T_02'] # +15 град С
        self.T_03 = vals['T_03'] # +50 град С

        "Шаг по времени:"
        self.time_step = vals['time_step'] # 10 нСек

        "Исходные данные газодиномических параметров:"
        self.v_pm = 770 # дульная скорость снаряда, м/с
        self.p_max = 290 # максимальное давление в канале ствола, Па

        "Производные параметры, получаемые в результате обработки исходных данных обратной задачи:"
        self.omega = None  # масса порохового заряда, кг
        self.W_0 = None  # объем зарядной каморы, м^3
        self.S = None  # площадь поперечного сечения канала ствола, м^2
        self.l_0 = None  # приведенная длина камеры, м
        self.psi_s = None  # значение функции газоприхода в момент распада зерна для семиканального пороха
        self.dzeta = None  # относительная масса воспламенительного состава
        self.phi = None  # коэффициент фиктивности массы снаряда
        self.R_g = None  # индивидуальная газовая постоянная, Дж\(кг*К)
        self.S_W0 = None  # начальная площадь поверхности теплоотдачи, м^2
        self.l_m = None  # длина ведущей части канала ствола, м
        self.I_e = None #
        self.f_ign = None #
        self.b_ign = None #

        """Вызов метода вычисления производных параметров:"""
        self.derived_parameters()

    def derived_parameters(self):
        """Вычисление производных параметров:"""
        self.omega = self.q*self.omega_q
        self.W_0 = self.omega / self.Delta
        self.S  = self.n_s*pi*self.d**2/ 4
        self.l_0 = self.W_0/self.S
        self.psi_s = self.kappa_1*(1+self.lambda_1)
        self.dzeta = (1 / self.Delta - 1 / self.delta) / (self.f_n / self.p_ign + self.b)
        self.phi = self.K+1/3*self.omega*(1+self.dzeta)/self.q
        self.R_g = self.f_n/self.T_v
        self.S_W0 = 4*self.W_0/self.d
        self.l_m = self.l_m_d*self.d
        self.I_e = self.I_en
        self.f_ign = self.f_n
        self.b_ign = self.b



class Point_Substitution(Cannon_System_Parameters):
    """Класс реализации математической модели нульмерной подстановки"""

    def __init__(self):
        super().__init__()
        self.dx_dt = None
        self.dt = self.time_step



    def f(self,T_0: float)->float:
        """Значение силы пороха:"""
        return self.f_n*(1+self.k_f*(T_0-self.T_ref))

    def get_I_e(self, T_0:float):
        """Значение импульса пороха:"""
        return self.I_en*(1-self.k_I*(T_0-self.T_ref))

    def p_a(self, x):
        """Давление воздушного столба перед снарядом:"""
        return self.p_a0*(1+(self.k_a*(self.k_a+1)/4)*
                          (x[1]/self.c_a0)**2+self.k_a*(x[1]/self.c_a0)*
                          (1+(((self.k_a+1)/4)*(x[1]/self.c_a0)**2))**(1/2))

    def psi(self, z):
        """Расчет значений функций газоприхода"""
        return (self.kappa_1*z*(1+self.lambda_1*z)*np.heaviside(1-z, 0.5)
                +(self.psi_s+self.kappa_2*(z-1)*(1+self.lambda_2*(z-1)))*np.heaviside(z-1, 0.5))

    def p_m(self, x, T_0:float)->float:
        """Основное уравнение внутренней баллистики:"""
        return (self.f(T_0)*self.omega*self.psi(x[5])+self.f_ign*self.omega*self.dzeta-
                (self.k-1)*((self.phi*self.q*x[1]**2/2)+x[2]+x[3]))/(self.S*x[0]-(self.omega/self.delta)*(1-self.psi(x[5]))
                    -self.omega*(self.psi(x[5])+self.dzeta)*self.b)

    def T(self, x, T_0: float) -> float:
        """Температура ОГПС:"""
        psi_val = self.psi(x[5])

        # Защита от деления на ноль
        if psi_val == 0:
            return self.T_02  # возвращаем нормальную температуру

        term = 1 / (psi_val + self.dzeta) * (x[0] / (self.l_0*self.Delta) - 1 / self.delta + psi_val / self.delta)
        return (self.p_m(x, T_0) / self.R_g) * (term - self.b)

    def mu_g(self, x, T_0:float)->float:
        """Коэффициент динамической вязкости (формула Сазерленда):"""
        return self.mu_g0*((self.T_0s+self.T_cs)/(self.T(x, T_0)+self.T_cs))*(self.T(x, T_0)/self.T_0s)**1.5

    def Re(self, x, T_0:float)->float:
        """Число Рейнольдса:"""
        return (self.omega*(1+self.dzeta)*x[1]*self.d)/(2*self.S*x[0]*self.mu_g(x, T_0))

    def Nus(self, x, T_0:float)->float:
        """Число Нуссельта:"""
        return 0.023*(self.Re(x, T_0)**0.8)*(self.Pr**0.4)

    def q_w(self, x, T_0:float)->float:
        """Плотность теплового потока от пороховых газов к стенкам канала ствола:"""
        T_gas = self.T(x, T_0)
        sqrt_term = max(0, x[4]) ** 0.5  # защита от отрицательного корня
        return self.Nus(x, T_0) * (self.lambda_g / self.d) * (T_gas - T_0 - sqrt_term)

    def l_p(self, x):
        """Путь, пройденный снарядом по каналу ствола:"""
        return x[0]-self.l_0

    def S_w(self, x):
        """Площадь поверхности теплоотдачи:"""
        return self.S_W0+pi*self.d*self.l_p(x)

    def point_substitution(self, t, x, T_0:float)-> ndarray[tuple[int], dtype[float64]]:
        """Реализация 'нульмерной' подастановки путем численного интегрирования системы ОДУ:"""
        self.dx_dt=np.zeros(6)
        self.dx_dt[0]=x[1] # координата снаряда, x_p
        self.dx_dt[1]=((((self.p_m(x, T_0)-self.p_a(x))*self.S)/(self.phi*self.q))
                       *np.heaviside(self.p_m(x, T_0)-self.p_a(x)-self.p_0, 0.5)) # скорость снаряда, v_p
        self.dx_dt[2]=self.p_a(x)*self.S*x[1] # работа сил сопротивления давления воздушного столба, A_pa
        self.dx_dt[3]=self.S_w(x)*self.q_w(x, T_0) # теплота горения пороха, Q_omega
        self.dx_dt[4]=((2*self.q_w(x, T_0)**2)/(self.c_b*self.rho_b*self.lambda_b))-(2*x[1]/x[0])*x[4] # термический КПД выстрела, teta_T
        self.dx_dt[5]=(self.p_m(x, T_0)/self.get_I_e(T_0))*np.heaviside(self.z_e-x[5], 0.5) # толщина горящего свода, z
        return self.dx_dt

    def in_con(self):
        """Описание начальных условий:"""
        return [self.l_0, 0, 0, 0, 0, 0]

    def time_step(self):
        """Присвоение шага по времени"""
        self.dt = self.time_step
        return self.dt

    def stop(self,t, x):
        """Условие остановки алгоритма:"""
        return self.l_0+self.l_m-x[0]

    def point_substitution_min(self, t, x):
        """Запись начальных параметров в систему ОДУ при T_0 = - 50 град C"""
        return self.point_substitution(t, x, self.T_01)

    def point_substitution_norm(self, t, x):
        """Запись начальных параметров в систему ОДУ при T_0 = + 15 град C"""
        return self.point_substitution(t, x, self.T_02)

    def point_substitution_max(self, t, x):
        """Запись начальных параметров в систему ОДУ при T_0 = + 50 град C"""
        return self.point_substitution(t, x, self.T_03)

    def report_min(self, t, x)-> list[int | Any]:
        """Список выходных параметров для функции RungeKutta4 в ODE_solvers при T_0 = - 50 град C"""
        return [self.psi(x[5]), self.p_m(x, self.T_01)/1e6, x[0] - self.l_0]

    def report_norm(self, t, x)-> list[int | Any]:
        """Список выходных параметров для функции RungeKutta4 в  ODE_solvers при T_0 = + 15 град C"""
        return [self.psi(x[5]), self.p_m(x, self.T_02) / 1e6, x[0] - self.l_0]

    def report_max(self, t, x)-> list[int | Any]:
        """Список выходных параметров  для функции RungeKutta4 в  ODE_solvers при T_0 = + 50 град C"""
        return [self.psi(x[5]), self.p_m(x, self.T_03) / 1e6, x[0] - self.l_0]



class Quasi_One_Dimensional_Substitution(Cannon_System_Parameters):
    """Класс реализации математической модели квазиодномерной (газодинамической) подстановки"""

    def __init__(self):
        super().__init__()
        self.x_p = None

    def conditions_for_lagrange(self, powder_name: str = 'ДГ-3 20/1', T_0: float = 288.15) -> dict:
        """Алгоритм, реализующий запись входных данных (conditions) для метода Лагранжа из библиотеки pyballistics:"""
        conditions = {
            'powders': [{'omega': self.omega, 'dbname': powder_name}],
            'init_conditions': {
                'q': self.q,
                'd': self.d,
                'W_0': self.W_0,
                'T_0': T_0,
                'phi_1': self.K,
                'p_0': self.p_0
            },
            'igniter': {
                'p_ign_0': self.p_ign,
                'k_ign': 1.25,
                'T_ign': 2427,
                'f_ign': 260000.0,
                'b_ign': 0.0006
            },
            'meta_lagrange': {
                'CFL': 0.9,
                'n_cells': 300
            },
            'stop_conditions': {
                'v_p': self.v_pm,
                'x_p': self.l_m
            }
        }

        return conditions

    def mean_pressure_lagrange(self, p_array, x_array):
        """Вычисление среднего баллистического давления для каждого момента времени:"""
        if p_array is None or x_array is None or len(p_array) == 0: # проверка на None
            return 0.0

        # Проверка и приведение размерностей
        if len(p_array) != len(x_array):
            # Если x_array длиннее (узлы сетки), центрируем
            if len(x_array) == len(p_array) + 1:
                x_centers = (x_array[:-1] + x_array[1:]) / 2 # центрирование узлов сетки
                x_array = x_centers
            else:
                # Обрезаем до минимальной длины
                min_len = min(len(p_array), len(x_array))
                p_array = p_array[:min_len]
                x_array = x_array[:min_len]

        self.x_p = x_array[-1] - x_array[0]
        if self.x_p <= 0:
            return np.mean(p_array)

        integral = np.trapezoid(p_array, x_array)
        return integral / self.x_p

    def search_transitional_points_lagrange(self, result, x_initial=0.0):
        """Запись значений параметров системы t, p_p, p_b, p_m и т.д. во всех точках пути"""
        layers = result['layers']

        t_vals = []
        p_b_vals = []
        p_p_vals = []
        p_m_vals = []
        v_p_vals = []
        l_p_vals = []
        z_vals = []
        psi_vals = []

        for layer in layers:
            x = layer['x']
            u = layer['u']
            p = layer['p']

            t_vals.append(layer['t'] * 1000)
            p_b_vals.append(p[0] / 1e6)
            p_p_vals.append(p[-1] / 1e6)
            p_m_vals.append(self.mean_pressure_lagrange(p, x) / 1e6)
            v_p_vals.append(u[-1])
            l_p_vals.append(x[-1] - x_initial)
            z_vals.append(layer.get('z_1', [0])[-1] if 'z_1' in layer else 0)
            psi_vals.append(layer.get('psi_1', [0])[-1] if 'psi_1' in layer else 0)

        t_vals = np.array(t_vals)
        p_b_vals = np.array(p_b_vals)
        p_p_vals = np.array(p_p_vals)
        p_m_vals = np.array(p_m_vals)
        v_p_vals = np.array(v_p_vals)
        l_p_vals = np.array(l_p_vals)
        z_vals = np.array(z_vals)
        psi_vals = np.array(psi_vals)
        points = {}

        # 0 - начало движения
        idx_0 = np.where(v_p_vals > 0.1)[0]
        if len(idx_0) > 0:
            i = idx_0[0]
            points['0'] = {'t': t_vals[i], 'p_b': p_b_vals[i], 'p_p': p_p_vals[i],
                           'p_m': p_m_vals[i], 'v_p': v_p_vals[i], 'l_p': l_p_vals[i]}

        # p_b_max
        idx = np.argmax(p_b_vals)
        points['p_b_max'] = {'t': t_vals[idx], 'p_b': p_b_vals[idx], 'p_p': p_p_vals[idx],
                             'p_m': p_m_vals[idx], 'v_p': v_p_vals[idx], 'l_p': l_p_vals[idx]}

        # p_p_max
        idx = np.argmax(p_p_vals)
        points['p_p_max'] = {'t': t_vals[idx], 'p_b': p_b_vals[idx], 'p_p': p_p_vals[idx],
                             'p_m': p_m_vals[idx], 'v_p': v_p_vals[idx], 'l_p': l_p_vals[idx]}

        # p_m_max
        idx = np.argmax(p_m_vals)
        points['p_m_max'] = {'t': t_vals[idx], 'p_b': p_b_vals[idx], 'p_p': p_p_vals[idx],
                             'p_m': p_m_vals[idx], 'v_p': v_p_vals[idx], 'l_p': l_p_vals[idx]}

        # e - окончание горения
        idx_e = np.where(psi_vals >= 0.99)[0]
        if len(idx_e) > 0:
            i = idx_e[0]
            points['e'] = {'t': t_vals[i], 'p_b': p_b_vals[i], 'p_p': p_p_vals[i],
                           'p_m': p_m_vals[i], 'v_p': v_p_vals[i], 'l_p': l_p_vals[i]}

        # m - вылет снаряда
        i = len(t_vals) - 1
        points['m'] = {'t': t_vals[i], 'p_b': p_b_vals[i], 'p_p': p_p_vals[i],
                       'p_m': p_m_vals[i], 'v_p': v_p_vals[i], 'l_p': l_p_vals[i]}

        return points, (t_vals, p_b_vals, p_p_vals, p_m_vals, v_p_vals, l_p_vals)




class Runge_Kutta_4(Point_Substitution):
    """Класс получения значений параметров в характерных точках при реализации "нульмерной" подстановки"""

    def __init__(self, max_steps: int = 10000):
        super().__init__()

        res_min = RungeKutta4(self.point_substitution_min, self.in_con(), self.stop, self.report_min, self.dt, 0, max_steps)
        res_norm = RungeKutta4(self.point_substitution_norm, self.in_con(), self.stop, self.report_norm, self.dt, 0, max_steps)
        res_max = RungeKutta4(self.point_substitution_max, self.in_con(), self.stop, self.report_max, self.dt, 0, max_steps)

        # ===== СОХРАНЯЕМ DataFrame КАК АТРИБУТЫ =====
        self.df_1 = pd.DataFrame(res_min, columns=['t', 'x_p', 'V_p', 'A_pa', 'Q_w', 'eta_t', 'z', 'psi', 'p_m', 'l_p'])
        self.df_2 = pd.DataFrame(res_norm,
                                 columns=['t', 'x_p', 'V_p', 'A_pa', 'Q_w', 'eta_t', 'z', 'psi', 'p_m', 'l_p'])
        self.df_3 = pd.DataFrame(res_max, columns=['t', 'x_p', 'V_p', 'A_pa', 'Q_w', 'eta_t', 'z', 'psi', 'p_m', 'l_p'])

        pd.set_option('display.precision', 5)

        # ===== ОБРАБОТКА РЕЗУЛЬТАТОВ ДЛЯ -50°C =====
        df_1 = pd.DataFrame(res_min, columns=['t', 'x_p', 'V_p', 'A_pa', 'Q_w', 'eta_t', 'z', 'psi', 'p_m', 'l_p'])
        pd.set_option('display.precision', 5)

        # 1. ТОЧКА НАЧАЛА ДВИЖЕНИЯ (0)
        df_1_0 = df_1.query('l_p>=0 and V_p>0')
        if len(df_1_0) > 0:
            index_0_1 = df_1_0[:1].index[0]
            t_01 = df_1.t.iloc[index_0_1] * 1e3
            t_0_1 = f'{t_01:.2f}'
            p_m_0_1 = f'{df_1.p_m.iloc[index_0_1]:.4f}'
            V_p_0_1 = f'{df_1.V_p.iloc[index_0_1]:.4f}'
            l_p_0_1 = f'{df_1.l_p.iloc[index_0_1]:.4f}'
        else:
            t_0_1 = '-'
            p_m_0_1 = '-'
            V_p_0_1 = '-'
            l_p_0_1 = '-'

        # 2. ТОЧКА МАКСИМАЛЬНОГО ДАВЛЕНИЯ (p_max)
        if len(df_1) > 0:
            max_index_1 = df_1['p_m'].idxmax()
            t_max1 = df_1.t.iloc[max_index_1] * 1e3
            t_max_1 = f'{t_max1:.2f}'
            p_m_max_1 = f'{df_1.p_m.iloc[max_index_1]:.4f}'
            V_p_max_1 = f'{df_1.V_p.iloc[max_index_1]:.4f}'
            l_p_max_1 = f'{df_1.l_p.iloc[max_index_1]:.4f}'
        else:
            t_max_1 = '-'
            p_m_max_1 = '-'
            V_p_max_1 = '-'
            l_p_max_1 = '-'

        # 3. ТОЧКА ОКОНЧАНИЯ ГОРЕНИЯ (e)
        df_e_1 = df_1[df_1['z'] >= self.z_e]  # >=, потому что z растет до z_e
        if len(df_e_1) > 0:
            e_index_1 = df_e_1.index[0]  # первый индекс, где достигли z_e
            t_e1 = df_1.t.iloc[e_index_1] * 1e3
            t_e_1 = f'{t_e1:.2f}'
            p_m_e_1 = f'{df_1.p_m.iloc[e_index_1]:.4f}'
            V_p_e_1 = f'{df_1.V_p.iloc[e_index_1]:.4f}'
            l_p_e_1 = f'{df_1.l_p.iloc[e_index_1]:.4f}'
        else:
            t_e_1 = '-'
            p_m_e_1 = '-'
            V_p_e_1 = '-'
            l_p_e_1 = '-'

        # 4. ТОЧКА ВЫЛЕТА СНАРЯДА (m)
        muzzle_1 = df_1[df_1['x_p'] >= (self.l_m + self.l_0)]
        if len(muzzle_1) > 0:
            index_m_1 = muzzle_1.index[0]
            t_m1 = df_1.t.iloc[index_m_1] * 1e3
            t_m_1 = f'{t_m1:.2f}'
            p_m_m_1 = f'{df_1.p_m.iloc[index_m_1]:.4f}'
            V_p_m_1 = f'{df_1.V_p.iloc[index_m_1]:.4f}'
            l_p_m_1 = f'{df_1.l_p.iloc[index_m_1]:.4f}'
        else:
            t_m_1 = '-'
            p_m_m_1 = '-'
            V_p_m_1 = '-'
            l_p_m_1 = '-'

        # ===== ВЫВОД ТАБЛИЦЫ =====
        res_T_01 = {
            'Точка': ['0', 'p_max', 'e', 'm'],
            't,мс': [t_0_1, t_max_1, t_e_1, t_m_1],
            'p_m,МПа': [p_m_0_1, p_m_max_1, p_m_e_1, p_m_m_1],
            'V_p,м/с': [V_p_0_1, V_p_max_1, V_p_e_1, V_p_m_1],
            'l_p,м': [l_p_0_1, l_p_max_1, l_p_e_1, l_p_m_1],
        }
        d_res_T_01 = pd.DataFrame(res_T_01)
        print('___ Результаты при температуре T = -50 градусов ___')
        print(d_res_T_01)

        # ===== ДЛЯ СОХРАНЕНИЯ В CSV =====
        res_T_0_1 = {
            't': [t_0_1, t_max_1, t_e_1, t_m_1],
            'p_m': [p_m_0_1, p_m_max_1, p_m_e_1, p_m_m_1],
            'v_p': [V_p_0_1, V_p_max_1, V_p_e_1, V_p_m_1],
            'l_p': [l_p_0_1, l_p_max_1, l_p_e_1, l_p_m_1],
        }
        index_labels = ['0', 'p_max', 'e', 'm']
        d_res_T_0_1 = pd.DataFrame(res_T_0_1, index=index_labels)
        d_res_T_0_1.to_csv('СМ6-62_Барышев_СА_output.csv', sep='\t', mode='w', float_format='%.4f', index=True)

        # ===== ОБРАБОТКА РЕЗУЛЬТАТОВ ДЛЯ +15°C =====
        df_2 = pd.DataFrame(res_norm, columns=['t', 'x_p', 'V_p', 'A_pa', 'Q_w', 'eta_t', 'z', 'psi', 'p_m', 'l_p'])
        pd.set_option('display.precision', 5)

        # 1. ТОЧКА НАЧАЛА ДВИЖЕНИЯ (0)
        df_2_0 = df_2.query('l_p>=0 and V_p>0')
        if len(df_2_0) > 0:
            index_0_2 = df_2_0[:1].index[0]
            t_02 = df_2.t.iloc[index_0_2] * 1e3
            t_0_2 = f'{t_02:.2f}'
            p_m_0_2 = f'{df_2.p_m.iloc[index_0_2]:.4f}'
            V_p_0_2 = f'{df_2.V_p.iloc[index_0_2]:.4f}'
            l_p_0_2 = f'{df_2.l_p.iloc[index_0_2]:.4f}'
        else:
            t_0_2 = '-'
            p_m_0_2 = '-'
            V_p_0_2 = '-'
            l_p_0_2 = '-'

        # 2. ТОЧКА МАКСИМАЛЬНОГО ДАВЛЕНИЯ (p_max)
        if len(df_2) > 0:
            max_index_2 = df_2['p_m'].idxmax()
            t_max2 = df_2.t.iloc[max_index_2] * 1e3
            t_max_2 = f'{t_max2:.2f}'
            p_m_max_2 = f'{df_2.p_m.iloc[max_index_2]:.4f}'
            V_p_max_2 = f'{df_2.V_p.iloc[max_index_2]:.4f}'
            l_p_max_2 = f'{df_2.l_p.iloc[max_index_2]:.4f}'
        else:
            t_max_2 = '-'
            p_m_max_2 = '-'
            V_p_max_2 = '-'
            l_p_max_2 = '-'

        # 3. ТОЧКА ОКОНЧАНИЯ ГОРЕНИЯ (e)
        df_e_2 = df_2[df_2['z'] >= self.z_e]  # >=, потому что z растет до z_e
        if len(df_e_2) > 0:
            e_index_2 = df_e_2.index[0]  # первый индекс, где достигли z_e
            t_e2 = df_2.t.iloc[e_index_2] * 1e3
            t_e_2 = f'{t_e2:.2f}'
            p_m_e_2 = f'{df_2.p_m.iloc[e_index_2]:.4f}'
            V_p_e_2 = f'{df_2.V_p.iloc[e_index_2]:.4f}'
            l_p_e_2 = f'{df_2.l_p.iloc[e_index_2]:.4f}'
        else:
            t_e_2 = '-'
            p_m_e_2 = '-'
            V_p_e_2 = '-'
            l_p_e_2 = '-'

        # 4. ТОЧКА ВЫЛЕТА СНАРЯДА (m)
        muzzle_2 = df_2[df_2['x_p'] >= (self.l_m + self.l_0)]
        if len(muzzle_2) > 0:
            index_m_2 = muzzle_2.index[0]
            t_m2 = df_2.t.iloc[index_m_2] * 1e3
            t_m_2 = f'{t_m2:.2f}'
            p_m_m_2 = f'{df_2.p_m.iloc[index_m_2]:.4f}'
            V_p_m_2 = f'{df_2.V_p.iloc[index_m_2]:.4f}'
            l_p_m_2 = f'{df_2.l_p.iloc[index_m_2]:.4f}'
        else:
            t_m_2 = '-'
            p_m_m_2 = '-'
            V_p_m_2 = '-'
            l_p_m_2 = '-'

        # ===== ВЫВОД ТАБЛИЦЫ =====
        res_T_02 = {
            'Точка': ['0', 'p_max', 'e', 'm'],
            't,мс': [t_0_2, t_max_2, t_e_2, t_m_2],
            'p_m,МПа': [p_m_0_2, p_m_max_2, p_m_e_2, p_m_m_2],
            'V_p,м/с': [V_p_0_2, V_p_max_2, V_p_e_2, V_p_m_2],
            'l_p,м': [l_p_0_2, l_p_max_2, l_p_e_2, l_p_m_2],
        }
        d_res_T_02 = pd.DataFrame(res_T_02)
        print('___ Результаты при температуре T = +15 градусов ___')
        print(d_res_T_02)

        # ===== ДЛЯ СОХРАНЕНИЯ В CSV =====
        res_T_0_2 = {
            't': [t_0_2, t_max_2, t_e_2, t_m_2],
            'p_m': [p_m_0_2, p_m_max_2, p_m_e_2, p_m_m_2],
            'v_p': [V_p_0_2, V_p_max_2, V_p_e_2, V_p_m_2],
            'l_p': [l_p_0_2, l_p_max_2, l_p_e_2, l_p_m_2],
        }
        index_labels = ['0', 'p_max', 'e', 'm']
        d_res_T_0_2 = pd.DataFrame(res_T_0_2, index=index_labels)
        d_res_T_0_2.to_csv('СМ6-62_Барышев_СА_output.csv', sep='\t', mode='a', float_format='%.4f', index=True)

        # ===== ОБРАБОТКА РЕЗУЛЬТАТОВ ДЛЯ +50°C =====
        df_3 = pd.DataFrame(res_max, columns=['t', 'x_p', 'V_p', 'A_pa', 'Q_w', 'eta_t', 'z', 'psi', 'p_m', 'l_p'])
        pd.set_option('display.precision', 5)

        # 1. ТОЧКА НАЧАЛА ДВИЖЕНИЯ (0)
        df_3_0 = df_3.query('l_p>=0 and V_p>0')
        if len(df_3_0) > 0:
            index_0_3 = df_3_0[:1].index[0]
            t_03 = df_3.t.iloc[index_0_3] * 1e3
            t_0_3 = f'{t_03:.2f}'
            p_m_0_3 = f'{df_3.p_m.iloc[index_0_3]:.4f}'
            V_p_0_3= f'{df_3.V_p.iloc[index_0_3]:.4f}'
            l_p_0_3 = f'{df_3.l_p.iloc[index_0_3]:.4f}'
        else:
            t_0_3 = '-'
            p_m_0_3 = '-'
            V_p_0_3 = '-'
            l_p_0_3 = '-'

        # 2. ТОЧКА МАКСИМАЛЬНОГО ДАВЛЕНИЯ (p_max)
        if len(df_3) > 0:
            max_index_3 = df_3['p_m'].idxmax()
            t_max3 = df_3.t.iloc[max_index_3] * 1e3
            t_max_3 = f'{t_max3:.2f}'
            p_m_max_3 = f'{df_3.p_m.iloc[max_index_3]:.4f}'
            V_p_max_3 = f'{df_3.V_p.iloc[max_index_3]:.4f}'
            l_p_max_3 = f'{df_3.l_p.iloc[max_index_3]:.4f}'
        else:
            t_max_3 = '-'
            p_m_max_3 = '-'
            V_p_max_3 = '-'
            l_p_max_3 = '-'

        # 3. ТОЧКА ОКОНЧАНИЯ ГОРЕНИЯ (e)
        df_e_3 = df_3[df_3['z'] >= self.z_e]  # >=, потому что z растет до z_e
        if len(df_e_3) > 0:
            e_index_3 = df_e_3.index[0]  # первый индекс, где достигли z_e
            t_e3 = df_3.t.iloc[e_index_3] * 1e3
            t_e_3 = f'{t_e3:.2f}'
            p_m_e_3 = f'{df_3.p_m.iloc[e_index_3]:.4f}'
            V_p_e_3 = f'{df_3.V_p.iloc[e_index_3]:.4f}'
            l_p_e_3 = f'{df_3.l_p.iloc[e_index_3]:.4f}'
        else:
            t_e_3 = '-'
            p_m_e_3 = '-'
            V_p_e_3 = '-'
            l_p_e_3 = '-'

        # 4. ТОЧКА ВЫЛЕТА СНАРЯДА (m)
        muzzle_3 = df_3[df_3['x_p'] >= (self.l_m + self.l_0)]
        if len(muzzle_3) > 0:
            index_m_3 = muzzle_3.index[0]
            t_m3 = df_3.t.iloc[index_m_3] * 1e3
            t_m_3 = f'{t_m3:.2f}'
            p_m_m_3 = f'{df_3.p_m.iloc[index_m_3]:.4f}'
            V_p_m_3 = f'{df_3.V_p.iloc[index_m_3]:.4f}'
            l_p_m_3 = f'{df_3.l_p.iloc[index_m_3]:.4f}'
        else:
            t_m_3 = '-'
            p_m_m_3 = '-'
            V_p_m_3 = '-'
            l_p_m_3 = '-'

        # ===== ВЫВОД ТАБЛИЦЫ =====
        res_T_03 = {
            'Точка': ['0', 'p_max', 'e', 'm'],
            't,мс': [t_0_3, t_max_3, t_e_3, t_m_3],
            'p_m,МПа': [p_m_0_3, p_m_max_3, p_m_e_3, p_m_m_3],
            'V_p,м/с': [V_p_0_3, V_p_max_3, V_p_e_3, V_p_m_3],
            'l_p,м': [l_p_0_3, l_p_max_3, l_p_e_3, l_p_m_3],
        }
        d_res_T_03 = pd.DataFrame(res_T_03)
        print('___ Результаты при температуре T = +50 градусов ___')
        print(d_res_T_03)

        # ===== ДЛЯ СОХРАНЕНИЯ В CSV =====
        res_T_0_3 = {
            't': [t_0_3, t_max_3, t_e_3, t_m_3],
            'p_m': [p_m_0_3, p_m_max_3, p_m_e_3, p_m_m_3],
            'v_p': [V_p_0_3, V_p_max_3, V_p_e_3, V_p_m_3],
            'l_p': [l_p_0_3, l_p_max_3, l_p_e_3, l_p_m_3],
        }
        index_labels = ['0', 'p_max', 'e', 'm']
        d_res_T_0_3 = pd.DataFrame(res_T_0_3, index=index_labels)
        d_res_T_0_3.to_csv('СМ6-62_Барышев_СА_output.csv', sep='\t', mode='a', float_format='%.4f', index=True)

        print("\n" + "-" * 65)
        print("Сравнение эталонных и расчетных значений дульной скорости")
        print("и максимального давления:")
        print("-" * 65)
        print('Расчетное максимальное давление = ', f'{df_2.p_m.iloc[max_index_2]:.4f}', 'МПа')
        print('Эталонное максимальное давление = ', self.p_max, 'МПа')
        epsilon_p = (df_2.p_m.iloc[max_index_2] - self.p_max) / self.p_max * 100
        print('Отклонение по давлению = ', f'{epsilon_p:.4f}', '%')
        print('Расчетное дульная скорость = ', f'{df_2.V_p.iloc[index_m_2]:.4f}', 'м/с')
        print('Эталонное дульная скорость = ', self.v_pm, 'м/с')
        epsilon_v = (df_2.V_p.iloc[index_m_2] - self.v_pm) / self.v_pm * 100
        print('Отклонение по скорости = ', f'{epsilon_v:.4f}', '%')



class Lagrange(Quasi_One_Dimensional_Substitution):
    """Класс для решения прямой задачи внутренней баллистики в квазиодномерной постановке в лагранжевых массовых координатах"""

    def __init__(self):
        super().__init__()

        conditions = self.conditions_for_lagrange()
        total = ozvb_lagrange(conditions)

        points, data = self.search_transitional_points_lagrange(total)
        t_vals, p_b_vals, p_p_vals, p_m_vals, v_p_vals, l_p_vals = data
        df = pd.DataFrame([
            {'Точка': f'«{k}»', 't, мс': f"{v['t']:.2f}",
             'p_b, МПа': f"{v['p_b']:.2f}", 'p_p, МПа': f"{v['p_p']:.2f}",
             'p_m, МПа': f"{v['p_m']:.2f}", 'v_p, м/с': f"{v['v_p']:.2f}",
             'l_p, м': f"{v['l_p']:.4f}"}
            for k, v in points.items()
        ])
        print(df.to_string(index=False))


class Plotter_Point_Substitution:
    """
    Класс для построения графиков результатов нульмерной подстановки

    Строит два графика:
    1. Зависимость параметров от времени для трех температур
    2. Зависимость параметров от пути снаряда для трех температур

    Требования:
    - Шрифт Times New Roman, размер 14
    - В легенде отображаются характерные точки
    """

    def __init__(self, solver: Runge_Kutta_4):
        """
        Инициализация с готовым решателем

        Параметры:
        ----------
        solver : Runge_Kutta_4
            Объект решателя с уже выполненными расчетами
        """
        # Используем данные из готового решателя
        self.df_1 = solver.df_1
        self.df_2 = solver.df_2
        self.df_3 = solver.df_3
        self.z_e = solver.z_e
        self.l_m = solver.l_m
        self.l_0 = solver.l_0

        self.temp_data = {
            '-50°C': {'df': self.df_1, 'color': 'blue', 'linestyle': '-', 'marker': 'o'},
            '+15°C': {'df': self.df_2, 'color': 'green', 'linestyle': '-', 'marker': 's'},
            '+50°C': {'df': self.df_3, 'color': 'red', 'linestyle': '-', 'marker': '^'}
        }

        self._setup_fonts()

    def _setup_fonts(self):
        """Настройка шрифтов для графиков"""
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14

    def _process_dataframe(self, df):
        """Обработка DataFrame для построения графиков"""
        df_processed = df.copy()
        df_processed['t_ms'] = df_processed['t'] * 1000

        points = {}

        # Начало движения
        moving = df_processed[df_processed['V_p'] > 0.1]
        if len(moving) > 0:
            idx0 = moving.index[0]
            points['0'] = {
                't': df_processed.loc[idx0, 't_ms'],
                'p': df_processed.loc[idx0, 'p_m'],
                'v': df_processed.loc[idx0, 'V_p'],
                'l': df_processed.loc[idx0, 'l_p']
            }

        # Максимальное давление
        if len(df_processed) > 0:
            idx_max = df_processed['p_m'].idxmax()
            points['p_max'] = {
                't': df_processed.loc[idx_max, 't_ms'],
                'p': df_processed.loc[idx_max, 'p_m'],
                'v': df_processed.loc[idx_max, 'V_p'],
                'l': df_processed.loc[idx_max, 'l_p']
            }

        # Конец горения
        burning = df_processed[df_processed['z'] >= self.z_e]
        if len(burning) > 0:
            idx_e = burning.index[0]
            points['e'] = {
                't': df_processed.loc[idx_e, 't_ms'],
                'p': df_processed.loc[idx_e, 'p_m'],
                'v': df_processed.loc[idx_e, 'V_p'],
                'l': df_processed.loc[idx_e, 'l_p']
            }

        # Вылет снаряда
        muzzle = df_processed[df_processed['x_p'] >= (self.l_m + self.l_0)]
        if len(muzzle) > 0:
            idx_m = muzzle.index[0]
            points['m'] = {
                't': df_processed.loc[idx_m, 't_ms'],
                'p': df_processed.loc[idx_m, 'p_m'],
                'v': df_processed.loc[idx_m, 'V_p'],
                'l': df_processed.loc[idx_m, 'l_p']
            }

        return {'df': df_processed, 'points': points}

    def _prepare_data(self):
        """Подготовка данных для построения графиков"""
        self.temp_data['-50°C']['processed'] = self._process_dataframe(self.df_1)
        self.temp_data['+15°C']['processed'] = self._process_dataframe(self.df_2)
        self.temp_data['+50°C']['processed'] = self._process_dataframe(self.df_3)

    def plot_pressure_velocity_vs_time(self, save_path: str = None, show_points: bool = True):
        """
        ГРАФИК 1: Зависимость p_m, v_p, l_p от времени для трех температур (нульмерная модель)
        """
        self._prepare_data()

        fig, ax1 = plt.subplots(figsize=(20, 10))

        ax1.set_xlabel('t, мс')
        ax1.set_ylabel('P, МПа', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Правая ось для скорости
        ax2 = ax1.twinx()
        ax2.set_ylabel('v_p, м/с', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Третья ось для пути
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('axes', 1.12))
        ax3.set_ylabel('l_p, м', color='black')
        ax3.tick_params(axis='y', labelcolor='black')

        colors = {'-50°C': 'royalblue', '+15°C': 'mediumseagreen', '+50°C': 'darkorange'}

        point_styles = {'0': 'o', 'p_max': 's', 'e': '^', 'm': 'D'}
        point_labels = {'0': '0 (начало движения)', 'p_max': 'p_max (макс. давление)',
                        'e': 'e (конец горения)', 'm': 'm (вылет снаряда)'}

        # Собираем все элементы для единой легенды
        handles = []
        labels = []

        for temp, data in self.temp_data.items():
            df_proc = data['processed']['df']
            points = data['processed']['points']
            color = colors[temp]

            # Давление (сплошная линия)
            line_p, = ax1.plot(df_proc['t_ms'], df_proc['p_m'],
                               color=color, linestyle='-',
                               linewidth=1, label=f'{temp}, p_m')
            handles.append(line_p)
            labels.append(f'{temp}, p_m')

            # Скорость (пунктирная линия)
            line_v, = ax2.plot(df_proc['t_ms'], df_proc['V_p'],
                               color=color, linestyle='--',
                               linewidth=1, label=f'{temp}, v_p')
            handles.append(line_v)
            labels.append(f'{temp}, v_p')

            # Путь (пунктирная линия с точками)
            line_l, = ax3.plot(df_proc['t_ms'], df_proc['l_p'],
                               color=color, linestyle='-.',
                               linewidth=1, label=f'{temp}, l_p')
            handles.append(line_l)
            labels.append(f'{temp}, l_p')

            if show_points:
                for point_name, point_data in points.items():
                    if point_name in point_styles:
                        # Точки на оси давления
                        ax1.scatter(point_data['t'], point_data['p'],
                                    color=color, marker=point_styles[point_name],
                                    s=100, zorder=5, alpha=0.8)
                        # Точки на оси скорости
                        ax2.scatter(point_data['t'], point_data['v'],
                                    color=color, marker=point_styles[point_name],
                                    s=100, zorder=5, alpha=0.8)
                        # Точки на оси пути
                        ax3.scatter(point_data['t'], point_data['l'],
                                    color=color, marker=point_styles[point_name],
                                    s=100, zorder=5, alpha=0.8)

        # Добавляем точки в легенду (один раз, с зеленым цветом как пример)
        for name in ['0', 'p_max', 'e', 'm']:
            handles.append(plt.Line2D([0], [0], marker=point_styles[name], color='w',
                                      markerfacecolor='green', markersize=14,
                                      label=point_labels[name]))

        # Единая легенда
        ax1.legend(handles, labels, loc='upper left', framealpha=0.9, fontsize=14)

        ax1.set_title(
            'Зависимость давления, скорости и пути снаряда от времени\nпри различных начальных температурахв нульмерной модели',
            fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()
        return fig

    def plot_pressure_velocity_vs_path(self, save_path: str = None, show_points: bool = True):
        """
        ГРАФИК 2: Зависимость p_m, v_p от пути снаряда для трех температур (нульмерная модель)
        """
        self._prepare_data()

        fig, ax1 = plt.subplots(figsize=(20, 10))

        ax1.set_xlabel('l_p, м')
        ax1.set_ylabel('P, МПа', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Правая ось для скорости
        ax2 = ax1.twinx()
        ax2.set_ylabel('v_p, м/с', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        colors = {'-50°C': 'royalblue', '+15°C': 'mediumseagreen', '+50°C': 'darkorange'}

        point_styles = {'0': 'o', 'p_max': 's', 'e': '^', 'm': 'D'}
        point_labels = {'0': '0 (начало движения)', 'p_max': 'p_max (макс. давление)',
                        'e': 'e (конец горения)', 'm': 'm (вылет снаряда)'}

        # Собираем все элементы для единой легенды
        handles = []
        labels = []

        for temp, data in self.temp_data.items():
            df_proc = data['processed']['df']
            points = data['processed']['points']
            color = colors[temp]

            # Давление (сплошная линия)
            line_p, = ax1.plot(df_proc['l_p'], df_proc['p_m'],
                               color=color, linestyle='-',
                               linewidth=1, label=f'{temp}, p_m')
            handles.append(line_p)
            labels.append(f'{temp}, p_m')

            # Скорость (пунктирная линия)
            line_v, = ax2.plot(df_proc['l_p'], df_proc['V_p'],
                               color=color, linestyle='--',
                               linewidth=1, label=f'{temp}, v_p')
            handles.append(line_v)
            labels.append(f'{temp}, v_p')

            if show_points:
                for point_name, point_data in points.items():
                    if point_name in point_styles:
                        ax1.scatter(point_data['l'], point_data['p'],
                                    color=color, marker=point_styles[point_name],
                                    s=100, zorder=5, alpha=0.8)
                        ax2.scatter(point_data['l'], point_data['v'],
                                    color=color, marker=point_styles[point_name],
                                    s=100, zorder=5, alpha=0.8)

        # Добавляем точки в легенду
        for name in ['0', 'p_max', 'e', 'm']:
            handles.append(plt.Line2D([0], [0], marker=point_styles[name], color='w',
                                      markerfacecolor='green', markersize=14,
                                      label=point_labels[name]))

        # Единая легенда
        ax1.legend(handles, labels, loc='lower center', framealpha=0.9, fontsize=14)

        ax1.set_title(
            'Зависимость давления и скорости снаряда от пройденного пути\nпри различных начальных температурах в нульмерной модели',
            fontsize=14, fontweight='bold')
        ax1.set_xlim(0, self.l_m)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()
        return fig

    def plot_all(self, save_dir: str = None):
        """Построение всех графиков"""


        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            time_path = f"{save_dir}/zero_pressure_velocity_time.png"
            path_path = f"{save_dir}/zero_pressure_velocity_path.png"
        else:
            time_path = None
            path_path = None

        self.plot_pressure_velocity_vs_time(save_path=time_path)
        self.plot_pressure_velocity_vs_path(save_path=path_path)

    def plot_all(self, save_dir: str = "results"):
        """
        Построение всех графиков с сохранением в папку


            Директория для сохранения графиков (по умолчанию "results")
        """
        import os

        # Создаем папку, если её нет
        os.makedirs(save_dir, exist_ok=True)

        # Сохранение графиков
        time_path = os.path.join(save_dir, "zero_model_pressure_velocity_time.png")
        path_path = os.path.join(save_dir, "zero_model_pressure_velocity_path.png")

        self.plot_pressure_velocity_vs_time(save_path=time_path)
        self.plot_pressure_velocity_vs_path(save_path=path_path)

        print(f"\n Графики сохранены в папку: '{save_dir}'")
        print(f"   - {time_path}")
        print(f"   - {path_path}")
        print("\n" + "=" * 65)
        print("ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ 'НУЛЬМЕРНОЙ' МОДЕЛИ ЗАВЕРШЕНО")
        print("=" * 65)



class Plotter_Quasi_One_Dimensional_Substitution:
    """
    Класс для построения графиков результатов квазиодномерной (газодинамической) подстановки
    """

    def __init__(self, lagrange_solver, zero_solver):
        """
        Инициализация с готовыми решателями

        Параметры:
        ----------
        lagrange_solver : Lagrange
            Объект решателя с выполненными газодинамическими расчетами
        zero_solver : Runge_Kutta_4
            Объект решателя с выполненными нульмерными расчетами
        """
        self.lagrange_solver = lagrange_solver
        self.zero_solver = zero_solver
        self.params = lagrange_solver

        # Сохраняем данные для T = +15°C
        self.lagrange_data = None
        self.zero_data = None

        self._setup_fonts()
        self._collect_data()

    def _setup_fonts(self):
        """Настройка шрифтов для графиков"""
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14

    def _collect_data(self):
        """Сбор данных для T = +15°C из обоих решателей"""
        from pyballistics import ozvb_lagrange

        # Данные из газодинамической модели для T = +15°C
        print("  Сбор данных из газодинамической модели для T = +15°C...")
        conditions = self._create_opts(self.params.T_02)
        total = ozvb_lagrange(conditions)
        self.lagrange_data = self._process_lagrange_result(total)

        # Данные из нульмерной модели для T = +15°C
        print("  Сбор данных из нульмерной модели для T = +15°C...")
        self.zero_data = self._process_zero_result(self.zero_solver.df_2)

    def _create_opts(self, T_0: float, powder_name: str = 'ДГ-3 20/1') -> dict:
        """Создание опций для ozvb_lagrange"""
        return {
            'powders': [{'omega': self.params.omega, 'dbname': powder_name}],
            'init_conditions': {
                'q': self.params.q,
                'd': self.params.d,
                'W_0': self.params.W_0,
                'T_0': T_0,
                'phi_1': self.params.K,
                'p_0': self.params.p_0
            },
            'igniter': {
                'p_ign_0': self.params.p_ign,
                'k_ign': 1.25,
                'T_ign': 2427,
                'f_ign': 260000.0,
                'b_ign': 0.0006
            },
            'meta_lagrange': {
                'CFL': 0.9,
                'n_cells': 300
            },
            'stop_conditions': {
                'v_p': self.params.v_pm,
                'x_p': self.params.l_m
            }
        }

    def _mean_pressure(self, p_array, x_array) -> float:
        """Вычисление среднего интегрального давления"""
        if p_array is None or x_array is None or len(p_array) == 0:
            return 0.0

        if len(p_array) != len(x_array):
            if len(x_array) == len(p_array) + 1:
                x_centers = (x_array[:-1] + x_array[1:]) / 2
                x_array = x_centers
            else:
                min_len = min(len(p_array), len(x_array))
                p_array = p_array[:min_len]
                x_array = x_array[:min_len]

        length = x_array[-1] - x_array[0]
        if length <= 0:
            return np.mean(p_array)

        integral = np.trapezoid(p_array, x_array)
        return integral / length

    def _process_lagrange_result(self, result):
        """Обработка результатов газодинамической модели"""
        layers = result['layers']

        t_vals = []
        p_b_vals = []
        p_p_vals = []
        p_m_vals = []
        v_p_vals = []
        l_p_vals = []
        z_vals = []
        psi_vals = []

        for layer in layers:
            x = layer['x']
            u = layer['u']
            p = layer['p']

            t_vals.append(layer['t'] * 1000)
            p_b_vals.append(p[0] / 1e6)
            p_p_vals.append(p[-1] / 1e6)
            p_m_vals.append(self._mean_pressure(p, x) / 1e6)
            v_p_vals.append(u[-1])
            l_p_vals.append(x[-1])

            z_val = layer.get('z_1', [0])
            if isinstance(z_val, (list, np.ndarray)):
                z_vals.append(z_val[-1] if len(z_val) > 0 else 0)
            else:
                z_vals.append(z_val)

            psi_val = layer.get('psi_1', [0])
            if isinstance(psi_val, (list, np.ndarray)):
                psi_vals.append(psi_val[-1] if len(psi_val) > 0 else 0)
            else:
                psi_vals.append(psi_val)

        t_vals = np.array(t_vals)
        p_b_vals = np.array(p_b_vals)
        p_p_vals = np.array(p_p_vals)
        p_m_vals = np.array(p_m_vals)
        v_p_vals = np.array(v_p_vals)
        l_p_vals = np.array(l_p_vals)
        z_vals = np.array(z_vals, dtype=float)
        psi_vals = np.array(psi_vals, dtype=float)

        points = self._find_characteristic_points(
            t_vals, p_b_vals, p_p_vals, p_m_vals, v_p_vals, l_p_vals, z_vals, psi_vals
        )

        return {
            't': t_vals, 'p_b': p_b_vals, 'p_p': p_p_vals, 'p_m': p_m_vals,
            'v_p': v_p_vals, 'l_p': l_p_vals, 'points': points
        }

    def _process_zero_result(self, df):
        """Обработка результатов нульмерной модели"""
        t_vals = df['t'].values * 1000
        p_m_vals = df['p_m'].values
        v_p_vals = df['V_p'].values
        l_p_vals = df['l_p'].values
        z_vals = df['z'].values
        psi_vals = df['psi'].values

        points = self._find_characteristic_points_zero(
            t_vals, p_m_vals, v_p_vals, l_p_vals, z_vals, psi_vals
        )

        return {
            't': t_vals, 'p_m': p_m_vals, 'v_p': v_p_vals, 'l_p': l_p_vals, 'points': points
        }

    def _find_characteristic_points(self, t, p_b, p_p, p_m, v_p, l_p, z, psi):
        """Нахождение характерных точек для газодинамической модели"""
        points = {}

        # 0 - начало движения
        idx_start = np.where(v_p > 0.1)[0]
        if len(idx_start) > 0:
            i = idx_start[0]
            points['0'] = {'t': t[i], 'p_m': p_m[i], 'v_p': v_p[i], 'l_p': l_p[i]}

        # p_m_max - максимум среднего давления
        if len(p_m) > 0:
            i = np.argmax(p_m)
            points['p_m_max'] = {'t': t[i], 'p_m': p_m[i], 'v_p': v_p[i], 'l_p': l_p[i]}

        # e - окончание горения
        idx_psi = np.where(psi >= 0.99)[0]
        if len(idx_psi) > 0:
            i = idx_psi[0]
            points['e'] = {'t': t[i], 'p_m': p_m[i], 'v_p': v_p[i], 'l_p': l_p[i]}

        # m - вылет снаряда
        i = len(t) - 1
        points['m'] = {'t': t[i], 'p_m': p_m[i], 'v_p': v_p[i], 'l_p': l_p[i]}

        return points

    def _find_characteristic_points_zero(self, t, p_m, v_p, l_p, z, psi):
        """Нахождение характерных точек для нульмерной модели"""
        points = {}

        idx_start = np.where(v_p > 0.1)[0]
        if len(idx_start) > 0:
            i = idx_start[0]
            points['0'] = {'t': t[i], 'p_m': p_m[i], 'v_p': v_p[i], 'l_p': l_p[i]}

        if len(p_m) > 0:
            i = np.argmax(p_m)
            points['p_m_max'] = {'t': t[i], 'p_m': p_m[i], 'v_p': v_p[i], 'l_p': l_p[i]}

        idx_psi = np.where(psi >= 0.99)[0]
        if len(idx_psi) > 0:
            i = idx_psi[0]
            points['e'] = {'t': t[i], 'p_m': p_m[i], 'v_p': v_p[i], 'l_p': l_p[i]}

        i = len(t) - 1
        points['m'] = {'t': t[i], 'p_m': p_m[i], 'v_p': v_p[i], 'l_p': l_p[i]}

        return points

    # ==================== ГРАФИК 1 ====================
    def plot_lagrange_full_vs_time(self, save_path: str = None):
        """
        ГРАФИК 1: Зависимость p_b, p_p, p_m, v_p, l_p от времени (газодинамика, T=+15°C)
        """
        data = self.lagrange_data
        points = data['points']

        fig, ax1 = plt.subplots(figsize=(20, 10))

        ax1.set_xlabel('t, мс')
        ax1.set_ylabel('P, МПа', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Давления
        line_pb, = ax1.plot(data['t'], data['p_b'], 'b-', linewidth=1, label='p_b (Давление на дно канала)')
        line_pp, = ax1.plot(data['t'], data['p_p'], 'c-', linewidth=1, label='p_p (Давление на дно снаряда)')
        line_pm, = ax1.plot(data['t'], data['p_m'], 'g-', linewidth=1, label='p_m (Среднее баллистическое давление)')

        # Скорость (правая ось)
        ax2 = ax1.twinx()
        ax2.set_ylabel('v_p, м/с', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        line_v, = ax2.plot(data['t'], data['v_p'], 'r--', linewidth=1, label='v_p')

        # Координата (третья ось)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('axes', 1.12))
        ax3.set_ylabel('l_p, м', color='black')
        ax3.tick_params(axis='y', labelcolor='black')
        line_l, = ax3.plot(data['t'], data['l_p'], 'm:', linewidth=1, label='l_p')

        # Характерные точки
        point_styles = {'0': 'o', 'p_m_max': 's', 'e': '^', 'm': 'D'}
        point_labels = {'0': '0 (начало движения)', 'p_m_max': 'p_m_max (макс. давление)',
                        'e': 'e (конец горения)', 'm': 'm (вылет снаряда)'}

        # Собираем ВСЕ элементы для ОДНОЙ легенды
        handles = [line_pb, line_pp, line_pm, line_v, line_l]
        labels = ['p_b (Давление на дно канала)', 'p_p (Давление на дно снаряда)', 'p_m (Среднее баллистическое давление)', 'v_p', 'l_p']

        # Добавляем точки в легенду
        for name in ['0', 'p_m_max', 'e', 'm']:
            if name in points:
                # Рисуем точки на соответствующих осях (для визуализации)
                ax1.scatter(points[name]['t'], points[name]['p_m'], color='green',
                            marker=point_styles[name], s=100, zorder=5)
                ax2.scatter(points[name]['t'], points[name]['v_p'], color='red',
                            marker=point_styles[name], s=100, zorder=5)
                ax3.scatter(points[name]['t'], points[name]['l_p'], color='magenta',
                            marker=point_styles[name], s=100, zorder=5)

                # Добавляем точку в легенду (один раз, с зеленым цветом)
                handles.append(plt.Line2D([0], [0], marker=point_styles[name], color='w',
                                          markerfacecolor='green', markersize=14,
                                          label=point_labels[name]))

        # Единая легенда (помещаем в правый нижний угол)
        ax1.legend(handles, labels, loc='lower right', framealpha=0.9, fontsize=14)

        ax1.set_title('Зависимость давлений на дно канала и снаряда, среднего баллистического давления, скорости снаряда\nи его координаты от времени при температуре T = +15°C в квазиодномерной модели', fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    # ==================== ГРАФИК 2 ====================
    def plot_lagrange_full_vs_path(self, save_path: str = None):
        """
        ГРАФИК 2: Зависимость p_b, p_p, p_m, v_p от пути (газодинамика, T=+15°C)
        """
        data = self.lagrange_data
        points = data['points']

        fig, ax1 = plt.subplots(figsize=(20, 10))

        ax1.set_xlabel('l_p, м')
        ax1.set_ylabel('P, МПа', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Давления
        line_pb, = ax1.plot(data['l_p'], data['p_b'], 'b-', linewidth=1, label='p_b (Давление на дно канала)')
        line_pp, = ax1.plot(data['l_p'], data['p_p'], 'c-', linewidth=1, label='p_p (Давление на дно снаряда)')
        line_pm, = ax1.plot(data['l_p'], data['p_m'], 'g-', linewidth=1, label='p_m (Среднее баллистическое давление)')

        # Скорость (правая ось)
        ax2 = ax1.twinx()
        ax2.set_ylabel('v_p, м/с', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        line_v, = ax2.plot(data['l_p'], data['v_p'], 'r--', linewidth=1, label='v_p')

        # Характерные точки
        point_styles = {'0': 'o', 'p_m_max': 's', 'e': '^', 'm': 'D'}
        point_labels = {'0': '0 (начало движения)', 'p_m_max': 'p_m_max (макс. давление)',
                        'e': 'e (конец горения)', 'm': 'm (вылет снаряда)'}

        # Собираем ВСЕ элементы для ОДНОЙ легенды
        handles = [line_pb, line_pp, line_pm, line_v]
        labels = ['p_b (Давление на дно канала)', 'p_p (Давление на дно снаряда)', 'p_m (Среднее баллистическое давление)', 'v_p']

        # Добавляем точки в легенду
        for name in ['0', 'p_m_max', 'e', 'm']:
            if name in points:
                # Рисуем точки на соответствующих осях
                ax1.scatter(points[name]['l_p'], points[name]['p_m'], color='green',
                            marker=point_styles[name], s=100, zorder=5)
                ax2.scatter(points[name]['l_p'], points[name]['v_p'], color='red',
                            marker=point_styles[name], s=100, zorder=5)

                # Добавляем точку в легенду (один раз, с зеленым цветом)
                handles.append(plt.Line2D([0], [0], marker=point_styles[name], color='w',
                                          markerfacecolor='green', markersize=14,
                                          label=point_labels[name]))

        # Единая легенда
        ax1.legend(handles, labels, loc='lower right', framealpha=0.9, fontsize=14)

        ax1.set_title('Зависимость давлений на дно канала и снаряда, среднего баллистического давления и скорости снаряда\nот пройденного пути при температуре T = +15°C в квазиодномерной модели', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, self.params.l_m)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    # ==================== ГРАФИК 3 ====================
    def plot_comparison_vs_time(self, save_path: str = None):
        """
        ГРАФИК 3: Сравнение газодинамической и нульмерной моделей от времени (T=+15°C)
        """
        lagrange = self.lagrange_data
        zero = self.zero_data
        points_lag = lagrange['points']
        points_zero = zero['points']

        fig, ax1 = plt.subplots(figsize=(20, 10))

        ax1.set_xlabel('t, мс')
        ax1.set_ylabel('P, МПа', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Давление (газодинамика и нульмерная)
        line_pm_lag, = ax1.plot(lagrange['t'], lagrange['p_m'], 'g-', linewidth=1, label='p_m (квазиодномерная модель)')
        line_pm_zero, = ax1.plot(zero['t'], zero['p_m'], 'g--', linewidth=1, label='p_m (нульмерная модель)')

        # Добавляем p_b и p_p для газодинамики
        line_pb, = ax1.plot(lagrange['t'], lagrange['p_b'], 'b-', linewidth=1, label='p_b (квазиодномерная модель)')
        line_pp, = ax1.plot(lagrange['t'], lagrange['p_p'], 'c-', linewidth=1, label='p_p (квазиодномерная модель)')

        # Скорость (правая ось)
        ax2 = ax1.twinx()
        ax2.set_ylabel('v_p, м/с', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        line_v_lag, = ax2.plot(lagrange['t'], lagrange['v_p'], 'r-', linewidth=1, label='v_p (квазиодномерная модель)')
        line_v_zero, = ax2.plot(zero['t'], zero['v_p'], 'r--', linewidth=1, label='v_p (нульмерная модель)')

        # Координата (третья ось)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('axes', 1.12))
        ax3.set_ylabel('l_p, м', color='black')
        ax3.tick_params(axis='y', labelcolor='black')
        line_l_lag, = ax3.plot(lagrange['t'], lagrange['l_p'], 'm-', linewidth=1, label='l_p (квазиодномерная модель)')
        line_l_zero, = ax3.plot(zero['t'], zero['l_p'], 'm--', linewidth=1, label='l_p (нульмерная модель)')

        # Характерные точки
        point_styles = {'0': 'o', 'p_m_max': 's', 'e': '^', 'm': 'D'}
        point_labels = {'0': '0 (начало)', 'p_m_max': 'p_m_max (макс. давление)',
                        'e': 'e (конец горения)', 'm': 'm (вылет)'}

        # Собираем ВСЕ элементы для ОДНОЙ легенды
        handles = [line_pb, line_pp, line_pm_lag, line_pm_zero, line_v_lag, line_v_zero, line_l_lag, line_l_zero]
        labels = ['p_b (квазиодномерная модель)', 'p_p (квазиодномерная модель)', 'p_m (квазиодномерная модель)', 'p_m (нульмерная модель)',
                  'v_p (квазиодномерная модель)', 'v_p (квазиодномерная модель)', 'l_p (квазиодномерная модель)', 'l_p (нульмерная модель)']

        # Рисуем точки и добавляем их в легенду
        for name in ['0', 'p_m_max', 'e', 'm']:
            if name in points_lag:
                # Точки газодинамики
                ax1.scatter(points_lag[name]['t'], points_lag[name]['p_m'], color='darkgreen',
                            marker=point_styles[name], s=100, zorder=5)
                ax2.scatter(points_lag[name]['t'], points_lag[name]['v_p'], color='darkred',
                            marker=point_styles[name], s=100, zorder=5)
                ax3.scatter(points_lag[name]['t'], points_lag[name]['l_p'], color='darkmagenta',
                            marker=point_styles[name], s=100, zorder=5)

                # Точки нульмерной
                if name in points_zero:
                    ax1.scatter(points_zero[name]['t'], points_zero[name]['p_m'], color='lightgreen',
                                marker=point_styles[name], s=80, zorder=5)
                    ax2.scatter(points_zero[name]['t'], points_zero[name]['v_p'], color='lightcoral',
                                marker=point_styles[name], s=80, zorder=5)
                    ax3.scatter(points_zero[name]['t'], points_zero[name]['l_p'], color='plum',
                                marker=point_styles[name], s=80, zorder=5)

                # Добавляем точку в легенду (для газодинамики)
                handles.append(plt.Line2D([0], [0], marker=point_styles[name], color='w',
                                          markerfacecolor='darkgreen', markersize=14,
                                          label=point_labels[name] + ' (газодин.)'))
                # Добавляем точку нульмерной в легенду
                handles.append(plt.Line2D([0], [0], marker=point_styles[name], color='w',
                                          markerfacecolor='lightgreen', markersize=14,
                                          label=point_labels[name] + ' (нульм.)'))

        # Единая легенда
        ax1.legend(handles, labels, loc='upper left', framealpha=0.9, fontsize=14)

        ax1.set_title('Зависимость давлений на дно канала и снаряда, среднего баллистического давления, скорости снаряда\nи его координаты от времени при температуре T = +15°C в квазиодномерной и нульмерной модели', fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    # ==================== ГРАФИК 4 ====================
    def plot_comparison_vs_path(self, save_path: str = None):
        """
        ГРАФИК 4: Сравнение газодинамической и нульмерной моделей от пути (T=+15°C)
        """
        lagrange = self.lagrange_data
        zero = self.zero_data
        points_lag = lagrange['points']
        points_zero = zero['points']

        fig, ax1 = plt.subplots(figsize=(20, 10))

        ax1.set_xlabel('l_p, м')
        ax1.set_ylabel('P, МПа', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Давления
        line_pm_lag, = ax1.plot(lagrange['l_p'], lagrange['p_m'], 'g-', linewidth=1, label='p_m (квазиодномерная модель)')
        line_pm_zero, = ax1.plot(zero['l_p'], zero['p_m'], 'g--', linewidth=1, label='p_m (нульмерная модель)')
        line_pb, = ax1.plot(lagrange['l_p'], lagrange['p_b'], 'b-', linewidth=1, label='p_b (квазиодномерная модель)')
        line_pp, = ax1.plot(lagrange['l_p'], lagrange['p_p'], 'c-', linewidth=1, label='p_p (квазиодномерная модель)')

        # Скорость (правая ось)
        ax2 = ax1.twinx()
        ax2.set_ylabel('v_p, м/с', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        line_v_lag, = ax2.plot(lagrange['l_p'], lagrange['v_p'], 'r-', linewidth=1, label='v_p (квазиодномерная модель)')
        line_v_zero, = ax2.plot(zero['l_p'], zero['v_p'], 'r--', linewidth=1, label='v_p (нульмерная модель)')

        # Характерные точки
        point_styles = {'0': 'o', 'p_m_max': 's', 'e': '^', 'm': 'D'}
        point_labels = {'0': '0 (начало)', 'p_m_max': 'p_m_max (макс. давление)',
                        'e': 'e (конец горения)', 'm': 'm (вылет)'}

        # Собираем ВСЕ элементы для ОДНОЙ легенды
        handles = [line_pb, line_pp, line_pm_lag, line_pm_zero, line_v_lag, line_v_zero]
        labels = ['p_b (квазиодномерная модель)', 'p_p (квазиодномерная модель)', 'p_m (квазиодномерная модель)',
                  'p_m (нульмерная модель)', 'v_p (квазиодномерная модель)', 'v_p (нульмерная модель)']

        # Рисуем точки и добавляем их в легенду
        for name in ['0', 'p_m_max', 'e', 'm']:
            if name in points_lag:
                # Точки газодинамики
                ax1.scatter(points_lag[name]['l_p'], points_lag[name]['p_m'], color='darkgreen',
                            marker=point_styles[name], s=100, zorder=5)
                ax2.scatter(points_lag[name]['l_p'], points_lag[name]['v_p'], color='darkred',
                            marker=point_styles[name], s=100, zorder=5)

                # Точки нульмерной
                if name in points_zero:
                    ax1.scatter(points_zero[name]['l_p'], points_zero[name]['p_m'], color='lightgreen',
                                marker=point_styles[name], s=80, zorder=5)
                    ax2.scatter(points_zero[name]['l_p'], points_zero[name]['v_p'], color='lightcoral',
                                marker=point_styles[name], s=80, zorder=5)

                # Добавляем точки в легенду
                handles.append(plt.Line2D([0], [0], marker=point_styles[name], color='w',
                                          markerfacecolor='darkgreen', markersize=14,
                                          label=point_labels[name] + ' (газодин.)'))
                handles.append(plt.Line2D([0], [0], marker=point_styles[name], color='w',
                                          markerfacecolor='lightgreen', markersize=14,
                                          label=point_labels[name] + ' (нульм.)'))

        # Единая легенда
        ax1.legend(handles, labels, loc='lower right', framealpha=0.9, fontsize=14)

        ax1.set_title('Зависимость давлений на дно канала и снаряда, среднего баллистического давления и скорости снаряда\nот пройденного пути при температуре T = +15°C в квазиодномерной и нульмерной модели', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, self.params.l_m)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    # ==================== ВСЕ ГРАФИКИ ====================
    def plot_all(self, save_dir: str = "results"):
        """
        Построение всех 4 графиков с сохранением в папку

        Параметры:
        ----------
        save_dir : str
            Директория для сохранения графиков (по умолчанию "results")
        """
        import os

        # Создаем папку, если её нет
        os.makedirs(save_dir, exist_ok=True)

        # График 1
        path1 = os.path.join(save_dir, "graph_1_lagrange_time.png")
        print("\n1. График: газодинамика, p_b, p_p, p_m, v_p, l_p от времени...")
        self.plot_lagrange_full_vs_time(save_path=path1)

        # График 2
        path2 = os.path.join(save_dir, "graph_2_lagrange_path.png")
        print("\n2. График: газодинамика, p_b, p_p, p_m, v_p от пути...")
        self.plot_lagrange_full_vs_path(save_path=path2)

        # График 3
        path3 = os.path.join(save_dir, "graph_3_comparison_time.png")
        print("\n3. График: сравнение моделей от времени...")
        self.plot_comparison_vs_time(save_path=path3)

        # График 4
        path4 = os.path.join(save_dir, "graph_4_comparison_path.png")
        print("\n4. График: сравнение моделей от пути...")
        self.plot_comparison_vs_path(save_path=path4)

        print(f"\n Все графики сохранены в папку: '{save_dir}'")
        print(f"   - {path1}")
        print(f"   - {path2}")
        print(f"   - {path3}")
        print(f"   - {path4}")
        print("\n" + "=" * 65)
        print("ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ КВАЗИОДНОМЕРНОЙ МОДЕЛИ ЗАВЕРШЕНО")
        print("=" * 65)






# ==================== ОСНОВНАЯ ЧАСТЬ ====================

# Расчет характерных точек для "нульмерного" подхода:
print("\n" + "=" * 65)
print("НАЧАЛО РАСЧЕТОВ ХАРАКТЕРНЫХ ТОЧЕК ДЛЯ 'НУЛЬМЕРНОГО' ПОДХОДА")
print("=" * 65)

# Создаем объект и запускаем расчеты
solver_1 = Runge_Kutta_4(max_steps=10000)

print("\n" + "=" * 65)
print("РАСЧЕТЫ ХАРАКТЕРНЫХ ТОЧЕК ДЛЯ 'НУЛЬМЕРНОГО' ПОДХОДА ЗАВЕРШЕНЫ")
print("Результаты сохранены в файл: СМ6-62_Барышев_СА_output.csv")
print("=" * 65)

# Расчет характерных точек для квазиодномерного подхода:
print("\n" + "=" * 65)
print("НАЧАЛО РАСЧЕТОВ ХАРАКТЕРНЫХ ТОЧЕК ДЛЯ КВАЗИОДНОМЕРНОГО ПОДХОДА")
print("=" * 65)

# Создаем объект и запускаем расчеты
solver_2 = Lagrange()

print("\n" + "=" * 65)
print("РАСЧЕТЫ ХАРАКТЕРНЫХ ТОЧЕК ДЛЯ КВАЗИОДНОМЕРНОГО ПОДХОДА ЗАВЕРШЕНЫ")
print("=" * 65)

# ==================== ПОСТРОЕНИЕ ГРАФИКОВ ====================

# Папка для сохранения всех графиков (относительный путь)
import os
SAVE_DIR = os.path.join(os.getcwd(), "results", "graphs")

# Создаем папку рекурсивно, если её нет
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📁 Папка для сохранения графиков: '{SAVE_DIR}'")

# Построение графиков для нульмерной модели
print("\n" + "=" * 65)
print("ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ НУЛЬМЕРНОЙ МОДЕЛИ")
print("=" * 65)

plotter_zero = Plotter_Point_Substitution(solver_1)
plotter_zero.plot_all(save_dir=SAVE_DIR)

# Построение графиков для квазиодномерной модели
print("\n" + "=" * 65)
print("ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ КВАЗИОДНОМЕРНОЙ МОДЕЛИ")
print("=" * 65)

plotter_quasi = Plotter_Quasi_One_Dimensional_Substitution(solver_2, solver_1)
plotter_quasi.plot_all(save_dir=SAVE_DIR)

print("\n" + "=" * 65)
print("🎉 ВСЕ РАСЧЕТЫ И ПОСТРОЕНИЕ ГРАФИКОВ УСПЕШНО ЗАВЕРШЕНЫ!")
print(f"📁 Результаты сохранены в папку: '{SAVE_DIR}'")
print("=" * 65)