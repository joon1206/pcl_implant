This is a placeholder for README

README should have the following: 
- explanations for each vairable, related math and concepts maybe
- quickstart example command
- inputs (volume, surface area, params JSON)
- outputs (CSV, plots; depends on direction I guess)
- model assumptions, link to docs if we have them
- how to cite (if it comes to that, but I doubt it)
- teammate introductions and bios
- END GOAL - PRODUCT!!!

add more if necessary. 
I have a repo structure written down in my notebook. 

## Degradation mathematical model as derived from first order chain scission model 
> Guys this is really fun I promise

We start off with degrdaation kinetics. We know the equations for the first order chain scission model and geometry dependence through surface area (SA), which is the following: \
$$\frac{dM_{n}}{dt} = -k_{d}M_{n}$$
$$M_{n}(t) = M_{n,0} e^{-k_{d}t}$$\
And hydrolysis (PCL degradation mechanism) is diffusion limited. The variables here are defined as: 
- $k_{d} = k_{0} (\frac{SA}{V})^m$
- $k_{0}$ is the material/environment constant. We only want the scaling relationship, so this being here is okay. But there should be ample literature on finding this value, especially for PCL.
- M is approximately 1 for surface controlled degradation.
- $\frac{SA}{V}$ is, as mentioned, surface area over volume.
Combine all of this together, and you get this beautiful equation:\
$$M_{n}(t) = M_{n,0} exp(-k_{0}(\frac{SA}{V})^{m}t)$$\
(Though this might need some double checking)
But essentially, this is a model for time-dependent molecular weight.

## Going from the time dependent degradation model to predicting mechanical properties
Since we are interested in the mechanical properties that change with degradation, we look at just that. 
For semicrystaline polymers, there is a power law relationship for calculating Young's modulus based on molecular wt, which goes like this: \
$$E(t)=E_{0} (\frac{M_{n}(t)}{M_{n,0}})^{\alpha}$$\
Where E is Young's modulus, $E_{0}$ is the original Young's modulus, and alpha is the the scaling exponent that is usually a value between 1 and 3 depending on the regime or solution type. We can substitute the previous expression for molecular wt to get a scaling relationship of Young's modulus with a set SA/V based on degredation time! The equation looks something like this: 
$$E(t) = E_{0} exp(- \alpha{} k_{0} (\frac{SA}{V})^{m} t)$$
Clearly, with this, we can get some nice graphs:
1. E vs t for different geometries (different SA/V values)
2. E vs SA/V at fixed times, which would be useful if we set a designated degradation timeline goal
3. Log-log plot for the power law assumption verification, but this can also be experimental if we can do that
4. Sensitivity plot for alpha and m scaling coefficients/exponents, compare to literature values.
5. Characteristic degradation time (optional); this would be $\frac{\ln{(f)}}{\alpha{}k_{0} (\frac{SA}{V})^{m}}$ where f is just the fraction of E(t) to its original E

So, to do this, we need lit values for m, alpha, k0, and the rest of the code to compure SA/V from the CAD file if it isn't there already. Plus E(t) curves. Please make sure that 1) the formulas actually make sense, and 2) populate the graphs and save them as PNGs and GIFs. The GIFs should show progression of mechanical strength decrease as degradation goes on (almost like a 3d plot as well, since we can do SA/V variations). (01/30)

Collaborators: Katie Cariaga.
I leave the rest to you. The basics for every graph has been done. You can find the rest in the \src folder. Feel free to clone this and work locally before making any changes. I'm going to keep the main branch to myself for now, just in case. If you want to work in C++ that is fine, but my (limited) experience in CS has taught me that people don't like C++ (cue, that image I sent you; remind me to send it if I didn't send it yet--I promise, it's really funny). I need you to do the following: 
1. Document format--this README is a mess, and my other documentation is even worse. So sorry. This includes making the math easier to read. I'm not the best at doing LaTeX in markdown, so it's kind of a mess. I would like to envision each "long" math formula to be its own line, similar to academic papers. That way it would be easier to read and understand!
2. Incomplete items, finished--this would be the rest of the computational modules. If there is math on constant shear degradation (I am sure there is ample literature on this, as polymer physics is pretty established as a field. If you would like, send me any papers so we can talk about this together. Really fun, no?
3. Literally anything else you would like to add. Anything that would increase efficiency is a plus. The way I do things, I usually end up focusing on the wrong details trying to make tiny things slightly faster, and I end up losing focus. So, I made it a point to almost "flow of consciousness" my way through this code base, which is why it's a pretty big mess. I would like some help on this.
\
\
\ 
Some additional comments made on 02/02/2026 \
Right now the E vs SA/V graoh takes time arguments as hours, which is why the difference is not as prominent. If you could either increase the "hour" cap by a crap ton, or even change it to years, it might be great. If you decide to do the latter, it would be mainly for PCL though, since PCL degradation takes really long. Other biodegradable polymers may not share the same longetivity in in-vitro conditions (namely, our second candidate, PLGA, which we scrapped early in the project). Please take some time to both digest the work done so far, and try out different values. To an extent, this is somewhat expected, since a higher SA/V means that there is a much higher surface area compared to volume, so under this model it degrades much faster. This is reflected in the E(t) changes over time, with lower SA/V numbers retaining a relatively high amount of its modulus, compared to its higher counterparts. \ \ 
Here's my question here, though. What is a typical SA/V ratio? If it is not necessary to explore "endpoint" ratios, then I would rather not include them in any of our graphics, since it would heavily skew the visuals. From what I'm seeing, there is clearly some "peak" forming, which means there is something to explore there. Think about it! 


\
\
\
\
\
THE JOHNS HOPKINS UNIVERSITY \
DEPARTMENT OF MATERIALS SCIENCE AND ENGINEERING \
WONJOON JEONG \
CONNIE WENG \
ZOE CHAN \
CELIA JUNG \
KATIA CARIAGA \
SOPHIA ADAMS \
DR. ORLA WILSON \
DR. HAI-QUAN MAO \
DR. MICHAEL ROBICH \
LAA DT
