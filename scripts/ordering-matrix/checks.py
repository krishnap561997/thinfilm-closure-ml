from common import *
import pickle
b1=pickle.load(open('b1.pkl','rb'))
# Benney limit of q-gauge tau_w closure: substitute q = h^3 + q1(Benney)
q1=sp.Rational(6,5)*Re*h**6*H[1]-cb*h**3*H[1]+sp.Symbol('Kcap')*h**3*H[3]/3
tw=3*q/h**2+b1['tw1']
sub={q:h**3+q1, Q[1]:dx(h**3)}
red=sp.expand((tw.subs(sub)) - 3*(h**3+q1)/h**2)
# keep first order: drop products of derivatives beyond first (q1*... in Re term)
lin=sp.Poly(red, *H[1:6]).as_expr()
print('tau_w closure - 3q/h^2 at q=h^3+q1 (first-order terms):', sp.factor(sp.expand(red).subs({H[1]**2:0})))
fam=pickle.load(open('fam.pkl','rb'))
for name,d in fam.items():
    hh,qq,A2=sp.symbols('hh qq AA',positive=True)
    # flux (q, A2, A3) with s from A2 = qq^2/hh * f2(s); Jacobian via implicit s
    f2=d['A2'].subs(s,sp.Symbol('ss')); f3=d['A3'].subs(s,sp.Symbol('ss')); ss=sp.Symbol('ss')
    A3=qq**3/hh**2*f3
    # ds/d(var) from A2 - qq^2/hh f2(ss)=0
    G=A2-qq**2/hh*f2
    ds={v:-sp.diff(G,v)/sp.diff(G,ss) for v in (hh,qq,A2)}
    row=[sp.diff(A3,v)+sp.diff(A3,ss)*ds[v] for v in (hh,qq,A2)]
    J=sp.Matrix([[0,1,0],[0,0,1],row])
    for sv in [0, 2, -2]:
        A2v=(qq**2/hh*f2).subs({ss:sv,hh:1,qq:1})
        Jn=J.subs({ss:sv,hh:1,qq:1,A2:A2v})
        ev=[complex(e) for e in sp.Matrix(Jn).evalf().eigenvals(multiple=True)]
        print(name,'s=',sv,'eigs:',[f'{e.real:.4f}{e.imag:+.4f}j' for e in ev])
    # realizability: min of A2 factor
    smin=sp.solve(sp.diff(d['A2'],s),s)[0]; print('  min h A2/q^2 =',sp.nsimplify(d['A2'].subs(s,smin)),'at s=',smin)
    sol=sp.solve(sp.Eq(d['A2'],sp.Symbol('X')),s); print('  s(X), X=h A2/q^2:',sol)
