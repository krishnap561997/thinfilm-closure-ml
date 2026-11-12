// Check the ML hook against Python: ml_check <ml.txt> <state.txt> <out.txt>
// state.txt: header "dx Re Ca theta ng n", then n lines "h q" (ghost cells included).
// out.txt:   header with column names, then one row per cell: every model input and output.
#include "ml/ml_closure.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: ml_check ml.txt state.txt out.txt\n"; return 1; }
    std::ifstream in(argv[2]);
    Parameters p; int n;
    if (!(in >> p.dx >> p.Re >> p.Ca >> p.theta >> p.ng >> n)) { std::cerr << "ml_check: bad header\n"; return 1; }
    p.We = p.Re * p.Ca;
    Field h(n), q(n);
    for (int i = 0; i < n; ++i) in >> h[i] >> q[i];
    MLClosure m(argv[1], p);
    std::cout << m.summary();
    const std::vector<std::string> names{"h_x", "q_x", "h_x_s0p5", "q_x_s0p5", "h_xxx_s0p5", "q_xx_s0p5", "kappa_x_We_s0p5"};
    const auto F = m.features(h, q, p, names);
    const auto O = m.outputs(h, q, p);
    std::ofstream out(argv[3]);
    out << std::setprecision(12);
    for (const auto& k : names) out << k << ' ';
    for (const auto& [k, v] : O) out << k << ' ';
    out << '\n';
    for (int i = 0; i < n; ++i) {
        for (const auto& k : names) out << F.at(k)[i] << ' ';
        for (const auto& [k, v] : O) out << v[i] << ' ';
        out << '\n';
    }
    return 0;
}
