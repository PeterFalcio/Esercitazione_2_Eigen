#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

//Per prima cosa scrivo le funzioni che mi servono per calacolare la soluzione dei sistemi

VectorXd solve_palu(const MatrixXd& A, const VectorXd& b) {
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}

VectorXd solve_qr(const MatrixXd& A, const VectorXd& b) {
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    return x;
}
//Questa è la funzione con la quale calcolo l'errore relativ'
double relative_error(const VectorXd& x_exact, const VectorXd& x_computed) {
    return (x_exact - x_computed).norm() / x_exact.norm();
}

int main() {
    // Define the systems
    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    VectorXd x_exact1(2);
    x_exact1 << -1.0e+0, -1.0e+00;

    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    VectorXd x_exact2(2);
    x_exact2 << -1.0e+0, -1.0e+00;

    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    VectorXd x_exact3(2);
    x_exact3 << -1.0e+0, -1.0e+00;

    // Solve each system using PALU and QR decomposition
    VectorXd x_palu1 = solve_palu(A1, b1);
    VectorXd x_qr1 = solve_qr(A1, b1);
    double error_palu1 = relative_error(x_exact1, x_palu1);
    double error_qr1 = relative_error(x_exact1, x_qr1);

    VectorXd x_palu2 = solve_palu(A2, b2);
    VectorXd x_qr2 = solve_qr(A2, b2);
    double error_palu2 = relative_error(x_exact2, x_palu2);
    double error_qr2 = relative_error(x_exact2, x_qr2);

    VectorXd x_palu3 = solve_palu(A3, b3);
    VectorXd x_qr3 = solve_qr(A3, b3);
    double error_palu3 = relative_error(x_exact3, x_palu3);
    double error_qr3 = relative_error(x_exact3, x_qr3);

    // Print results
    cout << "System 1:" << endl;
    cout << "PALU Solution: " << x_palu1.transpose() << endl;
    cout << "QR Solution: " << x_qr1.transpose() << endl;
    cout << "Relative Error (PALU): " << error_palu1 << endl;
    cout << "Relative Error (QR): " << error_qr1 << endl << endl;

    cout << "System 2:" << endl;
    cout << "PALU Solution: " << x_palu2.transpose() << endl;
    cout << "QR Solution: " << x_qr2.transpose() << endl;
    cout << "Relative Error (PALU): " << error_palu2 << endl;
    cout << "Relative Error (QR): " << error_qr2 << endl << endl;

    cout << "System 3:" << endl;
    cout << "PALU Solution: " << x_palu3.transpose() << endl;
    cout << "QR Solution: " << x_qr3.transpose() << endl;
    cout << "Relative Error (PALU): " << error_palu3 << endl;
    cout << "Relative Error (QR): " << error_qr3 << endl;

    return 0;
}
