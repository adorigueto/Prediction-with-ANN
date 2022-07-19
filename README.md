Made in the [COMPETENCE CENTER IN MANUFACTURING (CCM)](https://www.ccm.ita.br/), a laboratory of the [AERONAUTICS INSTITUTE OF TECHNOLOGY (ITA)](http://www.ita.br/).

# Overall
This dataset was produced within a Master's Dissertation work, consisting of two experiments---Exp1 and Exp2---based on turning an AISI H13 steel with cutting fluid. The first experiment with theoretically new-tool conditions produced 324 samples for each measured roughness parameter. The second with the cutting tool flank wear varying in three levels produced 288 samples for each measured roughness parameter.

At Exp1 and Exp2, the surface roughness was measured on six different spots after each machining run. Therefore, the other variables---including machining forces---are repeated five times at each condition. For example, the first six rows of `Exp1.csv` show the same condition (4), with varying roughness parameters (and roughness measurement position), and all the other values are repeated.

The factors and responses tracked in the files are:
- depth of cut (ap)
- cutting speed (vc)
- feed rate (f)
- arithmetic mean deviation (Ra)
- skewness (Rsk)
- kurtosis (Rku)
- mean width of profile elements (RSm)
- total height (Rt)
- cutting force (Fc)
- passive force (Fy)
- feed force (Fz)
- resultant force (F)
- tool condition (flank wear width) (TCond)

## Exp1
The first experiment---`Exp1.csv`---consisted of a full-factorial experiment with three factors varying in three levels (DoE: 3^3) and two replicas, summing up 54 machining runs.

The experiment's table is depicted below:

| *Factors* | *Symbol* | *Units* | *Levels* ||| *Number of levels* |
| --- | --- |
| Depth of cut | ap | mm | 0.25 | 0.5 | 0.8 | 3 |
| Cutting speed  | vc | m/min | 310 | 350 | 390 | 3 |
| Feed rate | f | mm/rev | 0.07 | 0.1 | 0.13 | 3 |

## Exp2
The second experiment---`Exp2.csv`---consisted of an experiment with three factors varying in different levels and two replicas, summing up 48 machining runs. In terms of flank wear width (VBB), the tool condition was considered a factor---TCond. This time the cutting speed was kept constant at 350 m/min.

The experiment's table is depicted below:

| *Factors* | *Symbol* | *Units* | *Levels* |||| *Number of levels* |
| --- | --- |
| Depth of cut | ap | mm | 0.5 | 0.8 ||| 2 |
| Feed rate | f | mm/rev | 0.07 | 0.09 | 0.11 | 0.13 | 4 |
| Tool condition | TCond | mm | 0.0 | 0.1 | 0.3 || 3 |

## Preparation phase for Experiment 2
The Exp2 used the tool condition as a factor, bringing the necessity of having the tool worn to different levels---new tool (VBB = 0.0 mm), mid-life tool (VBB = 0.1 mm), and end-of-life tool (VBB = 0.3 mm). Therefore, the cutting tools were prepared---worn---in `Prep.csv`.

The correspondent parameters of Exp2 were used with the correspondent cutting tools during the preparation phase.

This phase does not consist of a planned experiment. However, the surface roughness was collected (four times) at each tool wear measuring step.

# Insights
`Exp1.csv` and `Exp2.csv` were used to build ANN models to predict the arithmetic mean deviation (Ra). It can also be used to model the other measured roughness parameters, the machining forces, or the cutting tool wear. The `Prep.csv` was used in fast statistical analysis. In addition, `Exp1.csv` and `Exp2.csv` were used to statistically map the correlation between the experimental factors and the responses --- mainly by factorial ANOVA.

# Material and equipment
- Material: AISI H13 (mean hardness = 200 HV).
- Machine tool (CNC turning center): ROMI E280 (max rotation = 4k rpm, nominal power = 18.5 kW).
- Cutting tool: SandvikCoromant ISO TNMG 16 04 04-PF 4425; tool shank (holder): ISO MTJNL 2020K 16M1.
- Cutting fluid: a mixture of Blaser Swisslube Vasco 7000 with water in 8%. The acidity (pH) of the mixture was around 8.
- Roughness assessment: Mitutoyo portable roughness tester model Surftest SJ-210.
- Tool wear assessment: digital microscope Dino-Lite model AM4113ZT.
- Forces measurement: dynamometer Kistler Type 9265B, connected to a charge amplifier Kistler Type 5070 and an acquisition software Kistler Dynoware Type 2825A; one computer, and peripherical item: a highly insulated cable, Peripherical Component Interconnect (PCI interface), connection cable, and acquisition plate (A/D).

# Experiment scheme

![Experiment scheme](https://i.imgur.com/VWjzBJl.png)

# Acknowledgments
The author gives thanks to the Aeronautics Institute of Technology (ITA), the Competence Center in Manufacturing (CCM), and the Coordination for the Improvement of Higher Education Personnel (CAPES).