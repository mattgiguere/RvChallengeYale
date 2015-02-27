BeginPackage["GLSTools`"]

GLS::usage="GLS[t,y,ivar,freqlist] computes the generalized Lomb-Scargle periodogram for freqlist"
FAP::usage="FAP[t,y,ivar,freqlist,nmonte] returns a distribution of the false alarm probability"

Begin["`Private`"]

GLS[t_,y0_,ivar_,freqlist_]:=Module[{ivnorm},
  ivnorm = ivar/Total@ivar;
  GLS1[t,y0,ivnorm,#]& /@ freqlist
]

(* Zechmeister & Kurster, A&A 2009, 496 *)
GLS1[t_,y0_,ivar_,freq_] := Module[{ct,st,wy,y,c,s,yy,yc,ys,cch,cc,ss,cs,d},
  ct = Cos[freq*t]; 
  st = Sin[freq*t];
  wy = ivar*y0;
  y = Total[wy];
  c = Total[ivar*ct];
  s = Total[ivar*st];
  yy = Total[wy*y0] - y^2;
  yc = Total[wy*ct] - y*c;
  ys = Total[wy*st] - y*s;
  cch = Total[ivar*ct*ct];
  cc = cch - c*c;
  ss = (1-cch)-s*s;
  cs = Total[ivar*ct*st]-c*s;
  d = cc*ss-cs*cs;
  (ss*yc^2 + cc*ys^2 - 2*cs*yc*ys)/(yy*d)
]

FAP[t_,y0_,ivar_,freqlist_,nmonte_]:=Module[{pp1},
  pp1 = Table[GLS[RandomSample[t],y0,ivar,freqlist],{nmonte}];
  EmpiricalDistribution[Flatten@pp1]
]


End[]
EndPackage[]
