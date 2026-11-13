from common import *
phi=3*(eta-eta**2/2)
# candidate deformation shapes from the Re~1/eps expansion (normalised: psi(0)=0, psi'(1)=0, int psi=0, psi'(0)=1)
Sa=sp.expand(eta*(eta-2)*(4-10*eta+5*eta**2))
Sb=sp.expand(eta*(eta-2)*(-4-2*eta+29*eta**2-28*eta**3+7*eta**4))
res={}
for name,S0 in [('quartic (flux-disequilibrium mode)',Sa),('sextic (inertial mode)',Sb)]:
    psi=sp.expand(S0/sp.diff(S0,eta).subs(eta,0))
    assert sp.integrate(psi,(eta,0,1))==0 and sp.diff(psi,eta).subs(eta,1)==0 and psi.subs(eta,0)==0
    f=phi+s*psi
    A2=sp.factor(sp.integrate(f**2,(eta,0,1)))
    A3=sp.factor(sp.integrate(f**3,(eta,0,1)))
    D=sp.factor(sp.integrate(sp.diff(f,eta)**2,(eta,0,1)))
    us=sp.factor(f.subs(eta,1))
    print('====',name); print('psi =',psi)
    print('int f^2 =',sp.expand(A2)); print('int f^3 =',sp.expand(A3)); print('int fp^2 =',sp.expand(D)); print('f(1) =',us)
    res[name]=dict(psi=psi,A2=sp.expand(A2),A3=sp.expand(A3),D=sp.expand(D),us=us)
import pickle; pickle.dump(res,open('fam.pkl','wb'))

# NH pressure for semi-parabolic profile, inviscid pressureless leading-order time derivatives
u=sp.expand(3*q/h**3*(h*z-z**2/2)); w=w_of(u)
ht=-Q[1]; qt=-dx(sp.Rational(6,5)*q**2/h)
Dw=sp.expand(dt(w,ht,qt)+u*dx(w)+w*sp.diff(w,z))
I=sp.integrate(z*Dw,(z,0,h))
PNH=sp.expand(-dx(I))
inner=sp.factor(sp.expand(I))
print('\nint_0^h z Dw/Dt dz (semi-parabolic, inviscid LO) =');print(sp.collect(sp.expand(I),[Q[2],H[2]],sp.factor))
pickle.dump(dict(I=sp.expand(I)),open('nh.pkl','wb'))
# plug-flow check of formula: Serre: int z Dw/Dt = h^3/3 (U_x^2 - U_xt - U U_xx) with U=q/h
U=q/h; uu=U; ww=w_of(uu+0*z)
Dw2=sp.expand(dt(ww,ht,-dx(q**2/h))+uu*dx(ww)+ww*sp.diff(ww,z))
I2=sp.integrate(z*Dw2,(z,0,h))
Ut=dt(U,ht,-dx(q**2/h))
serre=h**3/3*(dx(U)**2-dx(Ut)-U*dx(dx(U)))
print('Serre check (should be 0):',sp.simplify(I2-serre))
