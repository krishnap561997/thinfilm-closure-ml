# Re ~ eps^-1 (delta ~ 1): q-gauge expansion, first order in derivative counting with Re retained
from common import *
Qt=sp.Symbol('Qt')
u0=sp.expand(3*q/h**3*(h*z-z**2/2))
w0=w_of(u0)
ht=-Q[1]
I0=sp.expand(sp.diff(u0,h)*ht+sp.diff(u0,q)*Qt + u0*dx(u0)+w0*sp.diff(u0,z))
Cap=sp.Symbol('Kcap')  # 1/Ca coefficient of h_xxx (order set by a)
F=sp.expand(-3-sp.diff(u0,z,2)+3*cb*H[1]-Cap*H[3]+Re*I0)
u1=solve_uzz(F)
cond=sp.integrate(u1,(z,0,h))
Qts=sp.solve(sp.Eq(cond,0),Qt)[0]
Qts=sp.expand(Qts)
print("Q_t (solvability = Galerkin model):")
print(sp.collect(Qts,[Q[1],H[1],H[3]],sp.factor))
u1s=sp.expand(u1.subs(Qt,Qts))
print("\nu1 =",sp.collect(u1s,[Re,cb,Cap],sp.factor))
print("int u1 =",sp.simplify(sp.integrate(u1s,(z,0,h))))
tw1=sp.expand(sp.diff(u1s,z).subs(z,0))
print("\ntau_w = 3q/h^2 +",sp.collect(tw1,[Re,cb,Cap],sp.factor))
A2c=sp.expand(2*sp.integrate(u0*u1s,(z,0,h)))
print("\n2 int u0 u1 =",sp.collect(A2c,[Re,cb,Cap],sp.factor))
# consistency: plain IBL q_t with tau_w closure and M=6q^2/5h vs Qts
qt_plain=sp.expand(-dx(sp.Rational(6,5)*q**2/h) + (3*h-(3*q/h**2+tw1))/Re - 3*cb/Re*h*H[1] + Cap/Re*h*H[3])
print("\nplain IBL (M=6q^2/5h, tau_w closed) minus Galerkin q_t:",sp.simplify(qt_plain-Qts))
qt_plain2=sp.expand(qt_plain-dx(A2c))
import pickle; pickle.dump(dict(u1=u1s,tw1=tw1,A2c=A2c,Qt=Qts,w0=w0),open('b1.pkl','wb'))
