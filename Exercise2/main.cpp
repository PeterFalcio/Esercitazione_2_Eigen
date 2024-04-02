#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

//Per prima cosa scrivo le funzioni che mi servono per calacolare la soluzione dei sistemi

VectorXd soluzione_PALU(const MatrixXd& A, const VectorXd& b) {
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}

VectorXd soluzione_QR(const MatrixXd& A, const VectorXd& b) {
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    return x;
}
//Questa è la funzione con la quale calcolo l'errore relativo
double errore_relativo(const VectorXd& x_esatta, const VectorXd& x_calcolata) {
    return (x_esatta - x_calcolata).norm() / x_esatta.norm();
}

int main() {
    // Definisco i sistemi
    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    VectorXd x_esatta1(2);
    x_esatta1 << -1.0e+0, -1.0e+00;

    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    VectorXd x_esatta2(2);
    x_esatta2 << -1.0e+0, -1.0e+00;

    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    VectorXd x_esatta3(2);
    x_esatta3 << -1.0e+0, -1.0e+00;

    // Risolvo i sistemi prima con la fattorizzazione PA = LU e poi con la fattorizzazione QR e per ciascuno dei due metodi
    // trovo l'errore relativo corrispondente
    VectorXd x_palu1 = soluzione_PALU(A1, b1);
    VectorXd x_qr1 = soluzione_QR(A1, b1);
    double error_palu1 = errore_relativo(x_esatta1, x_palu1);
    double error_qr1 = errore_relativo(x_esatta1, x_qr1);

    VectorXd x_palu2 = soluzione_PALU(A2, b2);
    VectorXd x_qr2 = soluzione_QR(A2, b2);
    double error_palu2 = errore_relativo(x_esatta2, x_palu2);
    double error_qr2 = errore_relativo(x_esatta2, x_qr2);

    VectorXd x_palu3 = soluzione_PALU(A3, b3);
    VectorXd x_qr3 = soluzione_QR(A3, b3);
    double error_palu3 = errore_relativo(x_esatta3, x_palu3);
    double error_qr3 = errore_relativo(x_esatta3, x_qr3);

    // Stampa dei risultati
    cout << "Sistema 1:" << endl;
    cout << "Soluzione PALU: " << x_palu1.transpose() << endl;
    cout << "Soluzione QR: " << x_qr1.transpose() << endl;
    cout << "Errore relativo (PALU): " << error_palu1 << endl;
    cout << "Errore relativo (QR): " << error_qr1 << endl << endl;

    cout << "Sistema 2:" << endl;
    cout << "Soluzione PALU " << x_palu2.transpose() << endl;
    cout << "Soluzione QR: " << x_qr2.transpose() << endl;
    cout << "Errore relativo (PALU): " << error_palu2 << endl;
    cout << "Errore relativo (QR): " << error_qr2 << endl << endl;

    cout << "Sistema 3:" << endl;
    cout << "Soluzione PALU: " << x_palu3.transpose() << endl;
    cout << "Soluzione QR: " << x_qr3.transpose() << endl;
    cout << "Errore relativo (PALU): " << error_palu3 << endl;
    cout << "Errore relativo (QR): " << error_qr3 << endl;

    return 0;
}
