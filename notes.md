

The original idea was from first order degradation kinetics from the chain scission model. $$ \frac{dM_n}{dt}=-k_d M_n $$ where $M_n (t) = M_{n,0}e^{-k_dt}$. We learned this in biomaterials I and various other chemistry/physical chemistry classes. The rationale behind using this was that hydrolysis is diffusion limited, and PCL degraded via hydrolysis. We defined the variables as:
- $k_d = k_0 (\frac{SA}{V})^m$
- $k_0$ here is the material/environment constant (aka diffusivity constant). If we only want the scaling relationship, then this being here won't affect the result too much. However, we want proper degradation timelines, which means this has to be found either through literature search or experimentally. In principal the way to actually measure this is pretty trivial, albeit "busy." 
- M is approximately 1 for a surface-controlled degradation. 
- $\frac{SA}{V}$ is the surface area over volume. 

Combine everything together, and you get this beautiful equation: $$ M_n(t) = M_{n,0} \exp{(-k_0(\frac{SA}{V})^mt)} $$
(Though this might need some double checking). Essentially, this is a model for the time-dependent molecular weight, based on an implant surface area to volume ratio, initial average polymer molecular weight, and the diffusivity constant. 


<span style="font-size:25px">Going from the time dependent degradation model to predicting mechanical properties</span>

Snce we are interested in the mechanical properties that change with degradation, we look at just that. 
For semicrystalline polymers, there is a power law relationship for calculating Young's modulus based on molecular wt, which goes like this: \
$$E(t)=E_{0} (\frac{M_{n}(t)}{M_{n,0}})^{\alpha}$$

Where E is Young's modulus, $E_{0}$ is the original Young's modulus, and alpha is the the scaling exponent that is usually a value between 1 and 3 depending on the regime or solution type. We can substitute the previous expression for molecular wt to get a scaling relationship of Young's modulus with a set SA/V based on degradation time! The equation looks something like this: 
$$E(t) = E_{0} \exp(- \alpha{} k_{0} (\frac{SA}{V})^{m} t)$$
Clearly, with this, we can get some nice graphs:
1. E vs t for different geometries (different SA/V values)
2. E vs SA/V at fixed times, which would be useful if we set a designated degradation timeline goal
3. Log-log plot for the power law assumption verification, but this can also be experimental if we can do that
4. Sensitivity plot for alpha and m scaling coefficients/exponents, compare to literature values.
5. Characteristic degradation time (optional); this would be $\frac{\ln{(f)}}{\alpha{}k_{0} (\frac{SA}{V})^{m}}$ where f is just the fraction of E(t) to its original E.

So, to do this, we need lit values for m, alpha, $k_0, and the rest of the code to compute SA/V from the CAD file if it isn't there already. Plus E(t) curves. Please make sure that 1) the formulas actually make sense, and 2) populate the graphs and save them as PNGs and GIFs. The GIFs should show progression of mechanical strength decrease as degradation goes on (almost like a 3d plot as well, since we can do SA/V variations). (01/30)

Collaborators: Katie Cariaga.
I leave the rest to you. The basics for every graph has been done. You can find the rest in the \src folder. Feel free to clone this and work locally before making any changes. I'm going to keep the main branch to myself for now, just in case. If you want to work in C++ that is fine, but my (limited) experience in CS has taught me that people don't like C++ (cue, that image I sent you; remind me to send it if I didn't send it yet--I promise, it's really funny). I need you to do the following: 
1. Document format--this README is a mess, and my other documentation is even worse. So sorry. This includes making the math easier to read. I'm not the best at doing LaTeX in markdown, so it's kind of a mess. I would like to envision each "long" math formula to be its own line, similar to academic papers. That way it would be easier to read and understand!
2. Incomplete items, finished--this would be the rest of the computational modules. If there is math on constant shear degradation (I am sure there is ample literature on this, as polymer physics is pretty established as a field. If you would like, send me any papers so we can talk about this together. Really fun, no?
3. Literally anything else you would like to add. Anything that would increase efficiency is a plus. The way I do things, I usually end up focusing on the wrong details trying to make tiny things slightly faster, and I end up losing focus. So, I made it a point to almost "flow of consciousness" my way through this code base, which is why it's a pretty big mess. I would like some help on this.

Some additional comments made on 02/02/2026 \
Right now the E vs SA/V graph takes time arguments as hours, which is why the difference is not as prominent. If you could either increase the "hour" cap by a crap ton, or even change it to years, it might be great. If you decide to do the latter, it would be mainly for PCL though, since PCL degradation takes really long. Other biodegradable polymers may not share the same longevity in in-vitro conditions (namely, our second candidate, PLGA, which we scrapped early in the project). Please take some time to both digest the work done so far, and try out different values. To an extent, this is somewhat expected, since a higher SA/V means that there is a much higher surface area compared to volume, so under this model it degrades much faster. This is reflected in the E(t) changes over time, with lower SA/V numbers retaining a relatively high amount of its modulus, compared to its higher counterparts. 

My question here, though. What is a typical SA/V ratio? If it is not necessary to explore "endpoint" ratios, then I would rather not include them in any of our graphics, since it would heavily skew the visuals. From what I'm seeing, there is clearly some "peak" forming, which means there is something to explore there. Think about it! 


<span style="font-size:25px">Mathematical "add-ons" for practical degradation model for implants</span>

Hydrolysis, while being a primary degradation mode, is not the only thing that affects the implant. There are many other things that are happening that can be added to the model to increase the practicality of the computational model. The goal here is to make it novel and useful, which means it should be more than a simple equation plotter. 

The few main modes of polymer degradation are surface erosion, bulk degradation, autocatalysis, enzymatic degradation, and cell-mediated degradation. The general consensus with PCL degradation is that the first phase is driven by random hydrolysis of ester groups within the polymer bulk, generally occurring over 1-2 years. This part is predominantly controlled by the availability of water molecules and the concentration of ester groups and carboxylic acid end groups. This is **pseudo-first order reaction kinetics**, assuming abundance of the governing agents. After the first phase, as a significant decrease in Mn (falling below 10% of the starting Mn) happens, along with an increase in crystallinity (50-80%). Upon a decrease in molecular wt below 10-8kDa, PCL is thought to enter the second degradation phase, exhibiting a linear mass loss through surface erosion, autocatalysis, and cell-mediated processes. 

We can include some stuff:
- Pseudo-first order molecular weight decay (we already kind of have this)
	- $M_n(t)=M_{n,0}e^{-kt}$
- Monomer/Acid product diffusion (this would be Fick's 2nd law, source term)
	- $\frac{\delta C_m}{\delta t}=\nabla \cdot (D_m\nabla C_m)+S(t)$
- Diffusivity increases with degradation (think porosity increase and damage proxy)
- Additionally, we could introduce continuous reaction-diffusion & chain scission distribution. I still need to hash out some math to figure this out, but if we could, we could predict MWD and PDI for a certain time frame at different implant geometries. This is A LOT of information to have, since local changes are hard to actually monitor consistently at an implant setting.
	- The reason this might be a little bit tricky is that they are semi-linear parabolic PDEs. What this means is that under finite time (which we are using), a boundary volume problem might explode in a singularity. I don't foresee a way to circumvent actually trying it out (i.e. brute forcing/trial-and-error ing it.)
	- Lucky for us, the solution is one-component. I have, however, not worked with anything that ISN'T in one spacial dimension in plane geometry (look up the Kolmogorov-Petrovsky-Piskunov equation, lovingly dubbed the KPP equation. The simplicity of the equation is often misleading,$\frac{\delta u}{\delta t}-\frac{\delta^2u}{\delta x^2}=F(u)$ where F(u) is a sufficiently smooth function with a few key properties. The wiki page will probably do a better job at explaining this than myself-https://en.wikipedia.org/wiki/KPP%E2%80%93Fisher_equation)
- There are already many models for hydrolytic degradation and diffusion. We can try to make a hydrolysis + oligomer/monomer/deg product diffusion + critical MW erosion onset (hence the comment on a "database connection," to take care of the "incubation period")
- The practical degradation model will include all of the above, if not more. 
- Some concerns on how to take into account the implausibility of a perfect sink condition (may not be a problem in a live human body, but will be for testing?)
	- For acidic products, we can use the Robin (mass transfer) boundary
	- $-D\nabla \cdot n = h(C_m - C_{m,\infty})$
	- So a larger h (it is h in the general expression, but for us, it would be $k_m$, the diffusivity) means stronger perfusion clearance, so **less autocatalysis**! 
	- Smaller h means acids are "trapped" which leads to strong autocatalysis!
	- Dr. Mao comment: ```glypase in regions are usually low. Enzyme--can ignore. very small factor.```



<span style="font-size:25px">Some notes from Dr. Mao regarding testing</span>
 - The testing itself
	 - Second level q, whether the drug is being released. next q you don't know. Is the device killing cells, or is the drug killing cells? 
	 - Drug release assay.
	 - Drug release kinetics are far more useful, since we know the drug works already. In-vivo we want to see inhibition, not killing tumor cells like PTX usually does. 
	 - In-vivo environment: no trug-injury triggers inflammation as a response. Inflammation cells are the target of the drug. So when you do, do fibroblasts, macrophage. 
	 - Doing cancer cells will not add anything meaningful. (we were not planning on this!)
	 - Drug release profile. Hemocompatibility test is unnecessary. If blood clots, it's actually good, blood is secondary. Dr. Mao's hunch is that "reporting mechanical requirements is good enough"
- Mechanical requirements:
	- Most you can do is accelerated test and extrapolate. Most for stability test. What device target product profile (need to load enough drug, with it also being loaded **uniformly** is important)
	- Uniform loading is very good to show. How do we do this? **Find a way to show this.** Either fluorescent, cut device into pieces and test separately, measure drug release options, HPLC, UV absorbance... HPLC is most common.
	- Check what wavelength you use. Check for purity also (any contaminants?)
	- AMount of drug--> UV is the most accessible. By the time we need it let Dr. Mao know. Useful for drug loading and release. Robustness, degradation stability. 
	- TPP TABLE!!! These are the most important properties to aim for. 
- Find data on needed amount of PTX for similar applications. Once you have general design you can get target . With this we can get a "how much we want this" profile.
- Most critical is the early phase immunoresponse. 3-4 or 6-88 weeks... we chose arbitrarily hahaha --> 6 weeks it is!
- Based on release curve, see how much is needed. Then reverse-math it to get total drug loading. Beyond high (10-15%) loading you'll see different release curves, since it is supersaturated. 
- This exercise will be helpful, since we'll know the **upper limit**
- **Sterilization concerns**
	- Once you make a device, if it is not clean, you can't sterilize. e-beam sterilization/dry-ice is the only way, but it is still difficult. But everything under sterile conditions will make it very expensive. 
	- Do we compromise? sterilization kills off a good % of the drug loaded. A solution to this--> just load more drug? Or Ethylene oxide gas sterilization?
	- But check if degradation/release kinetics/mechanical properties are all the same after... 
	- Prove if residual ethylene oxide will be a problem. 
	- ALL OF THIS IS LOW PRIORITY. Just drug release kinetics is more important. 
	- If we really want to we can do it in the hospital. E beam is tougher. Either way it is overkill. 
	- Analyze and describe in report, but you don't have to actually do. Fits our scope. 
	- Check out later. Don't waste too much time!
	- Have a clear release curve for real numbers 
- **Patenting and JHTV**
	- Application--> make one as a demo. Can cover a range of dimensions, to show that it works. 
	- Report of invention, check website
	- Write design report. LAwyers will help. INvention disclaim. WHat is the IP covering? DESIGN AND FORMULATION, most likely. 
	- The function of the patent is... need to cover a range of things. Dimensions. Polymer choice. Drug. What is NOT CHANGING is the device design, which is ours. 
	- Can maybe even get one for the applicator as well. If good on its own, cover applicator separately. 
	- Process of production (general description at least)
		- How to test locking stability. If not all the data, not yet. But file first, you can add later
	- Application utility
	- Do now so you have time to perfect this
	- Maybe call Dr. Robich separately, get an honest assessment. "If someone comes to sell this, would you use this?"
	- Side by side comparison to competitor products. 
		- PERFORMANCE PARAMETER!!! HOW FAST CAN THIS APPLY AND WHY IS IT BETTER!!
		- Ask Dr. Robich for more samples if possible. 
- **Overall TO DO**
	- Target product profile
	- Get samples to run competitive advantage tests, show how this implant is better than the others
	- Start clipper manufacturing. Make a BIG one for show. 
	- Team work together for JHTV, patent stuff... get report submission item list
	- Potential market analysis. Get a member on it
	- Who will be likely company that will be interested in this device? 

DISCORD MESSAGE (SENT)
1. Hey guys, this is an updated notes list (the .md file in resources, it's a markdown file so it should work for all OS types, don't be afraid) from some ideas I had and a short meeting Celia and I had w/ Dr. Mao about deg testing. There are a lot of other theoretical items about modeling as well, which I would like to work with Katie to complete. **Overall TO DO**
    
    - Target product profile (TPP) and potential market analysis; this would be getting fine details on competitor products (something like atriclip--try to find something that is EXTERNALLY applied (so any internal stents that would require **more** precise surgery would be a no go for comparison, something external that is used with an applicator would be great). It'd be something like cost comparisons, how long it takes to apply, the amount of effort and or sterilization, waste, etc., anything you can think of! This I will task @zoe chan
        
        - Get samples to run competitive advantage tests, show how this implant is better than the others @Celia Jung I think you're doing this already after Friday but all the trial stuff, PTX detection, the other items in the notes that apply to you
            
            - Start clipper manufacturing. Make a BIG one for show. (we talked about this. I'm sending another email to Materic, along with a "contact me" email. Could you compile a range of dimensions that this can be? I did ask you about the applicator already... that would be its own patentable product as well, according to Dr. Mao) @Connie
            - Team work together for JHTV, patent stuff... get report submission item list... @Sophia Adams Can you check the JHTV (JH tech venture) website and compile the actual items that we need to start a report of invention?
            - Who will be likely company that will be interested in this device? I think this is also @zoe chan
            
        
    
2. @Katie C I have some more computational items separately that we should work on, even if it's C++ I'm gonna ask you to solve the "Mathematical "add-ons" for practical degradation model for implants" portion of the .md file I just sent. All the equations that are needed are in the notes. This shouldn't take too long, since this is a multivariable equation solving task, and I would like you to have a codebase that visualizes molecular wt decay using all the equations given in the notes. If the Robin mass transfer boundary is too complicated, you can exclude that, since a properly circulating human body may be analogous to a perfect sink condition. After that is done, we'll combine it with the modulus equations and have a fully functioning correlative relationship. If you can have a UI for it (argument parsing, I assume C++ has something similar as well) that would also be wonderful, but for now that part is low priority. I'm going to work on this too.
    
3. Please respond to this msg if you've seen it!

///






























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
