(* ::Package:: *)

BeginPackage["PMC`"]

(* A Population Monte Carlo code for Gaussian Mixture Models. This is a direct implementation of arXiv:0903.0837 *)

buildModels::usage="buildModels[ll] translates a list of (wt, mu, covar) into a list of (wts,MultinormalDistribution). This is useful for later use in the code"

mkRandom::usage="mkRandom[model,n,fiddle] generates n random variates from the model. The model is assumed to be the output of buildModels. The fiddle option allows you
to fiddle with the output eg. to transform things etc"

iterPMC::usage="iterPMC[chi2,npop,model,eps,fiddle] iterates the PMC code for a Gaussian Mixture model. eps allows one to regularize the
covariance matrix generated, and prevent it from blowing up"

Begin["`Private`"]

Clear[buildModels,mkRandom];
buildModels[ll_]:=Module[{wt,models},
  wt = ll[[All,1]];
  wt = wt/Total[wt]; (* Normalize inside here *)
  models = MultinormalDistribution@@@ ll[[All,2;;3]];
  {wt,models}
];

mkRandom[ll_,n_,fiddle_]:= Module[{xx},
  xx=RandomVariate[MixtureDistribution@@ll, n];
  fiddle@@@xx
]

(* The major iteration code is below *)

iterPMC[chi2_, npop_, modelin_,eps_,fiddle_] :=
    Module[ {perplex,xx,lik,pps,totpp,wt,mix,reg,dets,model},
	(* The PMC loop starts here, execute this cell until the perplexity gets close to 1 *)
	(* Generate npop randoms and calculate the posterior at these points*)
	    model = modelin;
        perplex = 0;
        reg=1.0;
        While[perplex < 0.95,
         (* Generate random numbers *)
         xx = mkRandom[model, npop,fiddle];
         lik = chi2 @@@ xx;
         lik = lik - Min[lik];
         lik = Exp[-lik/2];

         (* Now compute the PDF for each of the mixtures and weight*)
         pps = PDF[#, xx] & /@ model[[2]];
         pps = pps*model[[1]];
         totpp = Total[pps, 1];
         pps = #/totpp & /@ pps;
         
         (* Compute weights *)
         wt = lik/totpp;
         wt = wt/Total[wt];

         (* Perplexity *)
         perplex = Exp[-Total[wt Log[wt]]]/npop;

         (* Now do the update step *)
         mix = Module[ {a, mu, sig, dx, dx1},
                   a = Total[wt* #];
                   mu = Total[wt*#*xx, 1]/a;
                   dx = # - mu & /@ xx;
                   dx1 = dx*wt*#;
                   sig = (Transpose[dx1].dx)/a + reg*eps;
                   {a, mu, sig}
               ] & /@ pps;
         dets = Det[#] & /@ mix[[All, 3]];
         If[ (Length@Select[dets, # <= 0 &]) > 0,
             Print["Determinant Problem"];
             Break[]
         ];
         reg = reg/1.2;
         model = buildModels[mix];
         ];
        {perplex, xx, model}
    ]


End[]
EndPackage[]
