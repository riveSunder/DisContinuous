#

{:style="text-align:center;"}
![Scutium glider animation with variable dt](assets/dt_scutium.gif)
Animation corresponding to Figure 1 in the intention article. A small glider in the _Scutium gravidus_ set of Lenia CA rules becomes unstable with too small and too large step size.  
{:style="text-align:center;"}

{:style="text-align:center;"}
![Orbium glider animation with variable dt](assets/dt_orbium.gif)
Animation corresponding to Figure 2 in the intention article. A small Lenia glider, _Orbium_, becomes unstable with too large, but not too small, step size.  
{
{:style="text-align:center;"}


{:style="text-align:center;"}
![Scutium glider animation with variable dtype](assets/dtypes_scutium.gif)
Animation corresponding to Figure 3 in the intention article. A small glider in the _Scutium gravidus_ set of Lenia CA rules becomes unstable when using the higher-precision `torch.float64`double data type.  
{
{:style="text-align:center;"}

{:style="text-align:center;"}
![Orbium glider animation with variable dtype](assets/dtypes_orbium.gif)
Animation corresponding to Figure 4 in the intention article. A small Lenia glider in called _Orbium_ does not become unstable when using the higher-precision `torch.float64`double data type.  
{:style="text-align:center;"}


{:style="text-align:center;"}
![Scutium glider animation with kernel size](assets/kernel_sizes_scutium.gif)
Animation corresponding to Figure 5 in the intention article. A larger kernel size of radius 65, corresponding to smaller (better) spatial resolution, leads to instability in a small _Scutium gravidus_ Lenia glider after about 24 steps. The simulation with radius 2 kernel predictably vanishes immediately. 
{:style="text-align:center;"}

{:style="text-align:center;"}
![Orbium glider animation with kernel size](assets/kernel_sizes_orbium.gif)
Animation corresponding to Figure 6 in the intention article. A larger kernel size of radius 65, corresponding to smaller (better) spatial resolution, is well tolerated by a small Lenia glider, _Orbium_. The glider remains stable after 512 steps at the "native" kernel radius of 13 and the enlarged radius of 65.
{:style="text-align:center;"}

