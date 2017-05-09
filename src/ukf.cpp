#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2*n_aug_+1; ++i) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  P_.fill(0.0);
  P_(0,0) = 1.0;
  P_(1,1) = 1.0;
  P_(2,2) = 1.0;
  P_(3,3) = 1.0;
  P_(4,4) = 1.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  /*****************************************************************************
 *  Initialization
 ****************************************************************************/
  if (!is_initialized_) {
    x_ = VectorXd(5);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];
      x_ << rho * cos(phi),
          rho * sin(phi),
          sqrt(rho_dot * cos(phi) * rho_dot * cos(phi) + rho_dot * sin(phi) * rho_dot * sin(phi)),
          0,
          0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_ << meas_package.raw_measurements_[0],
          meas_package.raw_measurements_[1],
          0,
          0,
          0;
    }

    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  //compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  //cout << "dt = " << measurement_pack.timestamp_ << "  " << previous_timestamp_ << "  " << dt << endl;
  time_us_ = meas_package.timestamp_;

  // predict only if some time passed between two measurements
  if (dt > 0.001) {
    Prediction(dt);
  }

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }

  // print the output
  //cout << "x_ = " << x_ << endl;
  //cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // Generate sigma points

  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  VectorXd x_aug = VectorXd(7);
  MatrixXd P_aug = MatrixXd(7, 7);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i=0; i < n_aug_; i++) {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * A_aug.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A_aug.col(i);
  }

  //predict sigma points
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double psy = Xsig_aug(3, i);
    double psy_dot = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_psy_dot_dot = Xsig_aug(6, i);

    double px_pred, py_pred, v_pred, psy_pred, psy_dot_pred;

    //avoid division by zero
    if (psy_dot > 0.001) {
      px_pred = px + v/psy_dot * (sin(psy + psy_dot*delta_t) - sin(psy));
      py_pred = py + v/psy_dot * (-cos(psy + psy_dot*delta_t) + cos(psy));
    }
    else {
      px_pred = px + v*cos(psy)*delta_t;
      py_pred = py + v*sin(psy)*delta_t;
    }
    v_pred = v;
    psy_pred = psy + psy_dot*delta_t;
    psy_dot_pred = psy_dot;

    //add noise
    px_pred = px_pred + 0.5*delta_t*delta_t*cos(psy)*nu_a;
    py_pred = py_pred + 0.5*delta_t*delta_t*sin(psy)*nu_a;
    v_pred = v_pred + delta_t*nu_a;
    psy_pred = psy_pred + 0.5*delta_t*delta_t*nu_psy_dot_dot;
    psy_dot_pred = psy_dot_pred + delta_t*nu_psy_dot_dot;

    //write predicted sigma points into right column
    Xsig_pred_(0, i) = px_pred;
    Xsig_pred_(1, i) = py_pred;
    Xsig_pred_(2, i) = v_pred;
    Xsig_pred_(3, i) = psy_pred;
    Xsig_pred_(4, i) = psy_dot_pred;
  }

  //predict state mean
  x_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    x_ = x_ + weights_(i)*Xsig_pred_.col(i);
  }

  //predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //normalize angles
    while (x_diff(3) <= -M_PI) x_diff(3) += 2*M_PI;
    while (x_diff(3) > M_PI) x_diff(3) -= 2*M_PI;

    P_ = P_ + weights_(i)*x_diff*x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;
  VectorXd z_diff;
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    //write predicted sigma points into right column
    Zsig(0, i) = Xsig_pred_(0, i); //px
    Zsig(1, i) = Xsig_pred_(1, i); //py
  }

  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i)*z_diff*z_diff.transpose();
  }

  // add measurement noise
  MatrixXd R = MatrixXd(n_z,n_z);
  R.fill(0.0);
  R(0, 0) = std_laspx_*std_laspx_;
  R(1, 1) = std_laspy_*std_laspy_;

  S = S + R;


  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //normalize angles
    while (x_diff(3) <= -M_PI) x_diff(3) += 2*M_PI;
    while (x_diff(3) > M_PI) x_diff(3) -= 2*M_PI;

    z_diff = Zsig.col(i) - z_pred;

    Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  z_diff = meas_package.raw_measurements_ - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  //calculate NIS
  NIS_laser_ = z_diff.transpose()*S.inverse()*z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  VectorXd z_diff;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double psy = Xsig_pred_(3, i);

    //write predicted sigma points into right column
    Zsig(0, i) = sqrt(px*px + py*py); // rho
    Zsig(1, i) = atan2(py, px); //phi
    Zsig(2, i) = (px*cos(psy)*v + py*sin(psy)*v) / Zsig(0, i) ; //rho_dot
  }

  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    z_diff = Zsig.col(i) - z_pred;
    //normalize angles
    while (z_diff(1) <= -M_PI) z_diff(1) += 2*M_PI;
    while (z_diff(1) > M_PI) z_diff(1) -= 2*M_PI;

    S = S + weights_(i)*z_diff*z_diff.transpose();
  }

  // add measurement noise
  MatrixXd R = MatrixXd(n_z,n_z);
  R.fill(0.0);
  R(0, 0) = std_radr_*std_radr_;
  R(1, 1) = std_radphi_*std_radphi_;
  R(2, 2) = std_radrd_*std_radrd_;

  S = S + R;


  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //normalize angles
    while (x_diff(3) <= -M_PI) x_diff(3) += 2*M_PI;
    while (x_diff(3) > M_PI) x_diff(3) -= 2*M_PI;


    z_diff = Zsig.col(i) - z_pred;
    //normalize angles
    while (z_diff(1) <= -M_PI) z_diff(1) += 2*M_PI;
    while (z_diff(1) > M_PI) z_diff(1) -= 2*M_PI;

    Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  z_diff = meas_package.raw_measurements_ - z_pred;
  //normalize angles
  while (z_diff(1) <= -M_PI) z_diff(1) += 2*M_PI;
  while (z_diff(1) > M_PI) z_diff(1) -= 2*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  //calculate NIS
  NIS_radar_ = z_diff.transpose()*S.inverse()*z_diff;
}
