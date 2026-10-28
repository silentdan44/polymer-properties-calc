# polymer-properties-calc

Набросок Python-библиотеки для оценки свойств полимерных систем через уже
созданный `openmm.app.Simulation`. Для механического расчёта нужен OpenMM 8.3+
с поддержкой `computeCurrentPressure` у баростатов.

## Текущий интерфейс

```python
from polymer_properties_calc import (
    MechanicalPropertiesCalculator,
    SamplingProtocol,
)

calculator = MechanicalPropertiesCalculator(
    axis="x",
    relax_transverse_axes=True,
    strain_rate=1e-4,             # engineering strain / ps
    stretch_interval_steps=100,
    max_strain=0.1,
    elastic_strain_range=(0.0, 0.02),
    yield_method="first_deviation",
    yield_tolerance=0.05,
    transverse_pressure=1.0,      # bar
)

calculator.run(simulation)  # Растягивает ячейку и продвигает Simulation.
print(calculator.raw_data.values["axial_pressure"])
result = calculator.analyze()
print(result.value, result.unit, result.poisson_ratio)
print(result.yield_stress, result.yield_strain)
print(result.max_tensile_stress, result.strain_at_max_tensile_stress)
```

### Теплоёмкость из энергетических флуктуаций

```python
from polymer_properties_calc import HeatCapacityCalculator, SamplingProtocol

heat_capacity = HeatCapacityCalculator(
    temperature=300,
    ensemble="NPT",              # Cp; для Cv используйте NVT и pressure=None
    pressure=1.0,                 # bar
    protocol=SamplingProtocol(equilibration_steps=10_000, samples=2_000, steps_per_sample=20),
)
heat_capacity.run(simulation)
result = heat_capacity.analyze()
print(result.value, result.unit)  # массовая теплоёмкость, J/(g K)
print(result.molar_heat_capacity, result.molar_unit)
```

Для NVT используется флуктуация полной энергии, для NPT — энтальпии
`H = U + P V`. Нулевая задержка автокорреляции энергии/энтальпии даёт дисперсию,
из которой считается теплоёмкость; вся автокорреляционная функция и оценка
времени корреляции сохраняются в `result.details`.

### Теплопроводность Green–Kubo

OpenMM не предоставляет универсальный микроскопический тепловой поток для всех
силовых полей. Поэтому расчёт принимает функцию, которая для конкретной модели
возвращает полный векторный тепловой ток (с конвективной и силовой/вириальной
частями) в `kJ nm/(mol ps)`:

```python
from polymer_properties_calc import (
    GreenKuboThermalConductivityCalculator,
    SamplingProtocol,
)

# Реализуйте для используемого force field; это должен быть полный
# экстенсивный тепловой ток, а не плотность тока.
def heat_current_provider(simulation):
    return heat_current_xyz_kj_nm_per_mol_ps(simulation)

conductivity = GreenKuboThermalConductivityCalculator(
    temperature=300,
    heat_current_provider=heat_current_provider,
    protocol=SamplingProtocol(equilibration_steps=10_000, samples=10_000, steps_per_sample=10),
    max_lag_fraction=0.5,
)
conductivity.run(simulation)
result = conductivity.analyze()
print(result.value, result.unit)
print(result.details["conductivity_by_lag_W_m_K"])
```

Интеграл Green–Kubo рассчитывается по автокорреляции теплового тока; `value`
соответствует заданному времени отсечения, а массив зависимости результата от
времени отсечения позволяет проверить наличие плато. Корректность зависит от
реализации `heat_current_provider` и достаточной длины равновесной траектории.

### Непрерывный расчёт Tg

```python
from polymer_properties_calc import GlassTransitionCalculator

tg = GlassTransitionCalculator(
    temperature_range=(250, 500),  # K
    rate_k_per_ns=100,
    mode="cycles",                # cooling, heating, heating_cooling, cycles
    cycles=3,
    pressure=1.0,                  # bar
    equilibration_steps=10_000,
    temperature_update_interval_steps=100,
    sample_interval_steps=10_000,
)

tg.run(simulation)
result = tg.analyze()
print(result.value, result.unit)
print(result.details["per_pass"])

# По запросу: объёмный КТР ниже и выше Tg, в 1/K.
result_with_cte = tg.analyze(include_cte=True)
print(result_with_cte.cte_below_tg, result_with_cte.cte_above_tg)
```

В каждом проходе температура линейно меняется с заданной скоростью; OpenMM
получает обновлённую температуру каждые `temperature_update_interval_steps`,
а плотность записывается каждые `sample_interval_steps`. Для нескольких
циклов один цикл — это нагрев от минимума к максимуму и последующее охлаждение.
В результате есть Tg для каждого прохода, а также средние оценки отдельно для
нагревания и охлаждения. При `include_cte=True` дополнительно оценивается
объёмный КТР по наклону плотности ниже и выше Tg: `alpha_V = -(1/rho) d(rho)/dT`.

Механический протокол строит кривую осевого давления от деформации. Модуль
Юнга оценивается наклоном кривой напряжение–деформация только на интервале
`elastic_strain_range`; на том же интервале рассчитывается коэффициент Пуассона
как минус наклон поперечной деформации от осевой. Предел текучести определяется
по `yield_method="first_deviation"` как первое отклонение от экстраполированного
упругого участка не менее чем на `yield_tolerance` (доля напряжения на конце
упругого интервала). Альтернатива `yield_method="stress_drop"` отмечает
предшествующий локальный максимум, когда напряжение после него падает на
заданную долю. Если критерий не достигнут до `max_strain`, поля текучести будут
`None`. Максимальное растягивающее напряжение и деформация при нём всегда
относятся к конечной траектории расчёта. Напряжения в результате заданы в GPa.
В OpenMM используется
`MonteCarloAnisotropicBarostat.computeCurrentPressure`; его давление включает
кинетическую часть и оценку вириала через конечную разность.

Доступные калькуляторы: `MechanicalPropertiesCalculator`,
`GlassTransitionCalculator`, `DensityCalculator`, `HeatCapacityCalculator`,
`GreenKuboThermalConductivityCalculator` и `CTECalculator`
(`ThermalExpansionCalculator` — синоним объёмного КТР).

### Газотранспортные свойства мембран

Для мембран есть расчёт коэффициента диффузии пенетранта по наклону MSD,
растворимости методом Widom и проницаемости по модели растворение–диффузия
`P = D*S`. Каталог `LITERATURE_PENETRANTS` содержит параметры сайтов TraPPE
для CH₄, CO₂ и N₂ из опубликованных моделей.

В качестве протокола из литературы добавлен пример triptycene polyimide
Tanis et al. (2023): 338.5 K, смесь CH₄:N₂ = 2:1 при 65.5 bar, газовая
диффузионная траектория 15–20 ns. В самой статье для газа использованы
Vrabec 2CLJQ для N₂ и Yin–MacKerell пятисайтовая модель для CH₄; это не те же
модели, что параметры TraPPE в каталоге. Чтобы воспроизвести статью, сам
`Simulation` должен содержать систему с соответствующими параметрами и
равновесной загрузкой. Для самостоятельного добавления газа есть runner:

```python
from polymer_properties_calc import MembraneGasTransportRunner, MembraneGasTransportCalculator

loaded_simulation = MembraneGasTransportRunner(
    penetrant="CH4",
    molecule_count=20,
    seed=7,
    equilibration_steps=1_000_000,
).prepare(simulation)

transport = MembraneGasTransportCalculator.from_literature(
    loaded_simulation,
    penetrant="CH4",
    pressure_bar=1.0,  # внешнее парциальное давление, bar
)
transport.run(loaded_simulation)
result = transport.analyze()
print(result.diffusion_coefficient_m2_per_s)
```

OpenMM не позволяет добавлять частицы в уже созданный `Context`, поэтому
`prepare()` возвращает новую `Simulation` с клонированной исходной системой,
добавленными молекулами и скопированным состоянием исходного расчёта. Число
молекул задаётся явно: оно не выводится автоматически из давления. После
вставки нужна релаксация загруженного полимера; её длина зависит от материала.
Вставка поддерживается для систем с одной стандартной `NonbondedForce` и
правилом смешивания Лоренца–Бертло. Для OpenMM-силовых полей с пользовательской
парной функцией автоматическое смешивание пока не поддерживается.

Для газов используются готовые модели из литературы. Произвольный органический
пенетрант можно параметризовать через OpenFF (необязательные зависимости
`openff-toolkit` и `openff-interchange`):

```python
from polymer_properties_calc import PenetrantPreset, MembraneGasTransportRunner

benzene = PenetrantPreset.from_smiles("benzene", "c1ccccc1")
loaded_simulation = MembraneGasTransportRunner(benzene, molecule_count=10).prepare(simulation)
```

Установить дополнительные пакеты можно командой `pip install '.[openff]'`.

OpenFF параметры LJ и заряды переносятся в `NonbondedForce`; внутренние
гармонические связи, углы и торсионы добавляются к системе. Это предполагает
совместимые правила смешивания для OpenFF и силового поля полимера. OpenFF
виртуальные сайты пока не поддерживаются. Сейчас сохраняется одна
сгенерированная конформация молекулы; Widom результат для гибких молекул будет
оценкой для этой жёсткой конформации без усреднения по внутренней
конформационной статистике.

Для растворимости Widom runner снимает кадры с MD и вычисляет энергию случайной
вставки тестовой молекулы в OpenMM, включая настроенный в исходной системе
метод электростатики:

```python
from polymer_properties_calc import WidomInsertionRunner

widom = WidomInsertionRunner.from_literature(
    simulation,
    penetrant="CO2",
    frame_count=100,
    seed=7,
)
widom.run(simulation)
solubility = widom.analyze()
print(solubility.excess_chemical_potential_kj_per_mol)
print(solubility.solubility_mol_per_m3_pa)
```

Литературный профиль Papchenko et al. использует последние 5 ns траектории и
10⁶ вставок на газ при 298 K; runner распределяет их между выбранными кадрами
(для 100 кадров это 10 000 вставок на кадр). В другой работе для полимерного
NBR сходимость исследовали на множестве независимых конфигураций и вплоть до
10⁸ вставок на кадр, что показывает зависимость нужного объёма выборки от
полимера и газа. Поэтому следует смотреть сходимость результата по числу
кадров и вставок, а профиль из статьи считать стартовой настройкой, не
универсальной гарантией точности. Расчёт даёт бесконечно-разбавленную
растворимость; для высокой загрузки и набухания нужны итерационные MD–Widom,
GCMC или осмотический ансамбль.

Профиль задаёт длину выборки, а не полное воспроизведение силового поля
Papchenko et al.: в той работе для полимера и CH₄ использовался PCFF, для CO₂ —
COMPASS. Текущие газовые константы TraPPE — отдельный набор из литературы;
для численного воспроизведения статьи нужно передать систему и параметры газа
из её модели.

Для воспроизведения MSD-протокола статьи можно передать соответствующее
равновесное состояние с газом и использовать `TRIPTYCENE_POLYIMIDE_N2_CH4_2023`;
его экспериментально заданная длительность диффузии — 15–20 ns. Параметры
силового поля не подменяются пресетом автоматически.

Для стеклообразного полимера одной точки по давлению недостаточно для полной
изотермы сорбции. Если рассчитаны концентрации при нескольких давлениях,
можно подогнать dual-mode модель:

```python
from polymer_properties_calc import fit_dual_mode_sorption

fit = fit_dual_mode_sorption(
    pressures_bar=[1, 5, 10, 30, 70],
    concentrations_mol_m3=[...],
)
print(fit.k_d_mol_m3_bar, fit.c_h_mol_m3, fit.b_per_bar)
```

Для существующего загруженного бокса `MembraneGasTransportCalculator` может
по-прежнему оценивать секущую растворимость как `C/p` при заданном внешнем
давлении. Widom runner оценивает коэффициент Генри в бесконечно-разбавленном
пределе. Проницаемость вычисляется как `D*S` и выдаётся в Barrer.

Калькуляторы плотности, Tg и КТР ожидают периодическую систему с уже
настроенным `MonteCarloBarostat` или `MonteCarloAnisotropicBarostat`.
Механический калькулятор заменяет существующий изотропный/анизотропный
баростат на анизотропный. Если `relax_transverse_axes=True`, он поддерживает
заданное давление на двух поперечных осях; ось растяжения фиксирована. Если
параметр выключен, баростат используется только как измеритель давления и
не меняет размеры ячейки. Температура берётся из интегратора или существующего
баростата; при невозможности определить её можно передать явно.

### Геометрия пор и PSD через Zeo++

`PorePropertiesCalculator` запускает внешний исполняемый файл Zeo++ `network`
для периодически снимаемых кадров OpenMM. Для каждого кадра строится CIF
полной ячейки P1, затем Zeo++ рассчитывает крупнейшую включённую сферу (LCD),
крупнейшую свободную сферу (PLD), включённую сферу вдоль свободного пути
(Dif), канальную размерность, доступные поверхность/объём и PSD.

```python
from polymer_properties_calc import PorePropertiesCalculator

pores = PorePropertiesCalculator(
    zeo_executable="/path/to/zeo++/network",
    equilibration_steps=20_000,
    n_frames=20,
    sample_interval_steps=5_000,
    probe_radius_angstrom=0.0,
    monte_carlo_samples=50_000,
)
pores.run(simulation)
result = pores.analyze()
print(result.pore_limiting_diameter_angstrom)
print(result.largest_cavity_diameter_angstrom)
print(result.accessible_volume_fraction)
print(result.pore_diameter_bins_angstrom, result.pore_size_distribution)
```

В Zeo++ PLD соответствует крупнейшей сферической пробе, проходящей по
периодическому каналу; LCD — крупнейшей сфере внутри полости. PSD — гистограмма
распределения локальных диаметров пор; по умолчанию возвращается производная
гистограмма Zeo++ (`Derivative_dist`), а `raw_data` хранит массивы по кадрам.
Для анализа объёма/поверхности по умолчанию проба имеет нулевой радиус; задайте
радиус молекулы, например 1.86 Å для центра N₂, если нужна доступность для
конкретной пробы. Радиус канала и радиус измерительной пробы передаются
одинаковыми, как рекомендуется в документации Zeo++. Увеличение
`monte_carlo_samples` повышает статистическую точность и время расчёта.

Zeo++ устанавливается отдельно от этого Python-пакета. Он использует
элементные радиусы CCDC по умолчанию, а `-ha` включает высокоточную схему
Voronoi; для полимеров с пользовательскими радиусами атомов перед запуском
следует настроить радиусы Zeo++ для используемой модели. Это геометрический
анализ мгновенных конфигураций: результаты зависят от кадров, размера ячейки,
радиусов атомов и выбранной пробы, и сами по себе не являются оценкой
экспериментальной сорбционной пористости.
