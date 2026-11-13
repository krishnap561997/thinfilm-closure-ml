# Checks against Richard, Gisclon, Ruyer-Quil & Vila (2019), EJMB 76, 7-25
from common import *
cot,K=sp.symbols('cot Kc')   # cos(theta)/Fr^2 = 3 cot/Re ; capillary kappa/Fr^2 -> 1/We
# (1) their eqs (32),(33) with lambda=3, cos/Fr^2 = 3cot/Re  vs our tau_w1 (fixed h) and Benney q1
tau1 = Re*h*(sp.Rational(1,3)*9*h**3 - 3*cot/Re)*H[1]
U1   = h/3*Re*h*(sp.Rational(2,5)*9*h**3 - 3*cot/Re)*H[1]
print('their tau1(0):',sp.expand(tau1),'   ours: 3 Re h^4 h_x - 3 cot h h_x')
print('their q1 = h U1:',sp.expand(h*U1),'   ours (Benney): 6/5 Re h^6 h_x - cot h^3 h_x')
# (2) EIM form (their eq 22) rewritten as a q-equation == our row-1 Galerkin model?
qt=sp.Symbol('qt'); U=q/h
ht=-Q[1]
E=sp.Rational(3,5)*q**2/h          # hU^2/2 + E2, E2 = hU^2/10
F=sp.Rational(27,35)*q**3/h**2     # hU^3/2 + F2, F2 = 19/70 hU^3
Et=sp.diff(E,h)*ht+sp.diff(E,q)*qt
hyd = 3*cot/Re*(h*ht/1 + 0)        # d/dt(h^2/2)*c + d/dx(h^2 U)*c = c h U h_x  (shown analytically)
eq=sp.Eq(Et+dx(F) + 3*cot/Re*h*U*H[1], U/Re*(3*h-3*U/h) + K*h*U*H[3])
sol=sp.expand(sp.solve(eq,qt)[0])
ours=sp.expand(sp.Rational(5,6)*((3*h-3*q/h**2)/Re - 3*cot/Re*h*H[1] + K*h*H[3]) - sp.Rational(17,7)*q/h*Q[1] + sp.Rational(9,7)*q**2/h**2*H[1])
print('EIM(22) q_t - our row-1 Galerkin q_t =',sp.simplify(sol-ours))
# (3) their consistency condition (59): F1 = (1/9 - C1/10)(4/5) lambda^2 h^5 ; Lavalle C1=1
C1=sp.Symbol('C1'); F1=(sp.Rational(1,9)-C1/10)*sp.Rational(4,5)*9*h**5
print('F1(C1=1) =',F1.subs(C1,1),'  (status doc: 2*9 h^5/225 =',sp.Rational(18,225)*h**5,')')
