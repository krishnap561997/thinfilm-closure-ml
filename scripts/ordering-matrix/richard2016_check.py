# Checks against Richard, Ruyer-Quil & Vila (2016), JFM 804 (dimensionless, lambda=3, Nusselt units)
import sympy as sp, numpy as np
h,U,phi,s,Re=sp.symbols('h U phi s Re',positive=True)
lam=3
# (1) skewness: their (3.45)/(3.46) vs ours
print('<u\'^3>/U^3 at order 0: theirs -2/35 ; ours f3-3f2+2 at s=0 = -2/35  (see richard_compare.py)')
# (2) wall shear: theirs (5.10)-(5.11)  tau_w = lam h + (1+f1)(3U/h - lam h) - f2 (phi - lam^2 h^2/45),  f1=7/3, f2=315 U/(2 lam^2 h^3)
f1=sp.Rational(7,3); f2=sp.Rational(315,2)*U/(lam**2*h**3)
tw_R=lam*h+(1+f1)*(3*U/h-lam*h)-f2*(phi-lam**2*h**2/45)
# ours, quartic family: tau_w=(q/h^2)(3+s), q=hU, with h^3 phi = (q^2/h)(f2q(s)-1), f2q = 6/5 - 2s/35 + 2s^2/315
f2q=sp.Rational(6,5)-2*s/35+2*s**2/315
s_of=sp.solve(sp.Eq(h**4*phi/(h*U)**2, f2q-1),s)
s_lin=sp.series(s_of[0].subs(phi,sp.Symbol('p')+U**2/(5*h**2)),sp.Symbol('p'),0,2).removeO()  # linear in phi - phi_eq(U)
tw_Q=(U/h)*(3+s_lin.subs(sp.Symbol('p'),phi-U**2/(5*h**2)))
# linearise both about Nusselt (h=1,U=1,phi=1/5)
du,dp=sp.symbols('du dp')
lin=lambda e: sp.expand(sp.series(e.subs({h:1,U:1+du,phi:sp.Rational(1,5)+dp}).subs({du:sp.Symbol('t')*du,dp:sp.Symbol('t')*dp}),sp.Symbol('t'),0,2).removeO().subs(sp.Symbol('t'),1))
print('linearised tau_w, Richard 2016 :',lin(tw_R))
print('linearised tau_w, ours (quartic):',lin(tw_Q))
# (3) homogeneous relaxation (h=1, x-independent): U_t = (1/Re)(3h - tau_w);  their enstrophy eq (3.76): (h^2/2) phi_t = -(U/Re)[(2f1/3)(lam h-3U/h) + (2f2/3)(phi-phi_eq)]
Ut_R=lin((3*h-tw_R)); 
ph_R=lin(-2*U*((2*f1/3)*(lam*h-3*U/h)+(2*f2/3)*(phi-lam**2*h**2/45)))
# same model but skewness kept as a flux instead of converted (energy source = U(lam h-3U/h) only, eq 3.43):
ph_F=lin(-2*U*(f1*(lam*h-3*U/h)+f2*(phi-lam**2*h**2/45)))
def rates(a,b):
    M=sp.Matrix([[a.coeff(du),a.coeff(dp)],[b.coeff(du),b.coeff(dp)]])
    return sorted([round(float(-e),4) for e in M.eigenvals(multiple=True)])
print('decay rates (units 1/(Re h^2)): exact',[round(np.pi**2/4,4),round(9*np.pi**2/4,4)])
print('  Richard 2016 as published (skewness -> relaxation):',rates(Ut_R,ph_R))
print('  Richard 2016 with skewness kept as flux            :',rates(Ut_R,ph_F))
# (4) implied relaxation constant for the h^3 phi equation, at Nusselt: coefficient of (phi-phi_eq)
print('  enstrophy relaxation coefficient, published: -(4 U f2/3) at Nusselt =',sp.nsimplify((-4*U*f2/3).subs({U:1,h:1})),'  ours quartic: -35')
