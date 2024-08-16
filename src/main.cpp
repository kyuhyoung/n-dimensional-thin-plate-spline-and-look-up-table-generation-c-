#include <algorithm>
#include "opencv2/opencv.hpp"
//#include "Timer.h"
#include <time.h>
#include <iterator>
#include <random>

using namespace std;
using namespace cv;

bool is_only_number(const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(), 
	        s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}
tuple<vector<cv::Mat>, vector<vector<double> >, vector<double>, vector<double> > generate_src_tgt(int n_dim)
{
	std::vector<cv::Mat> li_mat_src;
	std::vector<std::vector<double> > li_li_tgt;
	int n_ctrl_pnt = n_dim * n_dim;
	cv::Mat mat_noise = cv::Mat::zeros(n_dim, n_ctrl_pnt, CV_64FC1);
	double from = 10, to = 100, mean = 0, stddev = 10.0 / 3.0;
	double pad = (to - from) / 20.0, interbal = (to - from) / 10;
	randn(mat_noise, Scalar(mean), Scalar(stddev));
	for(int iP = 0; iP < n_ctrl_pnt; iP++)
	{
		cv::Mat mat_src = cv::Mat::zeros(n_dim, 1, CV_64FC1);
		cv::randu(mat_src, Scalar(from), Scalar(to));
		li_mat_src.push_back(mat_src);
		std::vector<double> li_tgt(n_dim);
		cv::Mat mat_noised = mat_src + mat_noise.col(iP);
		for(int iD = 0; iD < n_dim; iD++)
		{
			li_tgt[iD] = mat_noised.at<double>(iD, 0); 
		}
		li_li_tgt.push_back(li_tgt);
	}
	vector<double> li_pad(n_dim), li_interv(n_dim);
	for(int iD = 0; iD < n_dim; iD++)
	{
		li_pad[iD] = pad;	li_interv[iD] = interbal; 
	}
	return tuple<vector<cv::Mat>, vector<vector<double> >, vector<double>, vector<double> >(li_mat_src, li_li_tgt, li_pad, li_interv);
}



bool compute_parameters(cv::Mat& param, const std::vector<cv::Mat>& li_warped, const std::vector<std::vector<double> >& li_ctrl)
//bool compute_parameters(cv::Mat& param, const std::vector<cv::Mat>& li_warped, std::vector<cv::Mat>& li_ctrl)
{
	int n_dim = li_warped[0].rows, num_points = li_warped.size();
	cv::Mat K = cv::Mat::zeros(num_points, num_points, CV_64FC1);
	for (int rr = 0; rr < num_points; rr++)
	{
		for (int cc = rr; cc < num_points; cc++)
		{
			double dist = MAX(0.0000000001, norm(li_warped[rr] - li_warped[cc]));
			K.at<double>(rr, cc) = dist;
			K.at<double>(cc, rr) = dist;
		}
	}
	cv::Mat P = cv::Mat::ones(num_points, n_dim + 1, CV_64FC1);
	for (int rr = 0; rr < num_points; rr++)
	{
		for (int iD = 0; iD < n_dim; iD++)
		{
			P.at<double>(rr, iD + 1) = li_warped[rr].at<double>(iD, 0);
		}
	}
	cv::Mat L_top;  hconcat(K, P, L_top);
	cv::Mat L_bot;  hconcat(P.t(), cv::Mat::zeros(n_dim + 1, n_dim + 1, CV_64FC1), L_bot);
	cv::Mat L;  vconcat(L_top, L_bot, L);
	cv::Mat t0; cv::invert(L, t0, cv::DECOMP_SVD);
	int n_pt_ctrl = li_ctrl.size();
	cv::Mat mat_ctrl = cv::Mat::zeros(n_pt_ctrl + n_dim + 1, n_dim, CV_64FC1);
	for (int iP = 0; iP < n_pt_ctrl; iP++)
	{
		for (int iD = 0; iD < n_dim; iD++)
		{
			//mat_ctrl.at<double>(iP, iD) = li_ctrl[iP].at<double>(iD, 0);
			mat_ctrl.at<double>(iP, iD) = li_ctrl[iP][iD];
		}
	}
	param = t0 * mat_ctrl;
	return true;
}


	std::pair<std::vector<double>, std::vector<double> > find_input_volume(const std::vector<cv::Mat>& li_p3d)
	{
		std::vector<double> minP, maxP;
		int n_pt = li_p3d.size();
		if (n_pt <= 0) return std::pair<std::vector<double>, std::vector<double> >(minP, maxP);
		int n_dim = li_p3d[0].rows;
		minP.resize(n_dim); maxP.resize(n_dim);
		for (int iD = 0; iD < n_dim; iD++)
		{
			//minP[iD] = std::numeric_limits<float>::max();   maxP[iD] = std::numeric_limits<float>::min();
			minP[iD] = 10000000000000000;   maxP[iD] = -1000000000000000;
		}
		for (int iP = 0; iP < n_pt; iP++)
		{
			for (int iD = 0; iD < n_dim; iD++)
			{
				if (li_p3d[iP].at<double>(iD, 0) > maxP[iD])
				{
					maxP[iD] = li_p3d[iP].at<double>(iD, 0);
				}
				if (li_p3d[iP].at<double>(iD, 0) < minP[iD])
				{
					minP[iD] = li_p3d[iP].at<double>(iD, 0);
				}
			}
		}
		return std::pair<std::vector<double>, std::vector<double> >(minP, maxP);
	}



	std::tuple<std::vector<int>, std::vector<int>, std::vector<cv::Mat> > generate_list_of_obj_to_warp(
		const std::pair<std::vector<double>, std::vector<double> >& pa_min_max_cm, const std::vector<double>& p_pad_cm, 
		const std::vector<double>& p_interval_cm)
	{
		std::vector<double> p_min_cm = pa_min_max_cm.first, p_max_cm = pa_min_max_cm.second;
		int n_dim = p_interval_cm.size();
		std::vector<double> from_cm(n_dim);
		for (int iD = 0; iD < n_dim; iD++)
		{
			from_cm[iD] = p_min_cm[iD] - p_pad_cm[iD];
			//std::cout << "iD : " << iD << " / " << n_dim << ", p_min_cm[iD] : " << p_min_cm[iD] << ", p_max_cm[iD] : " << p_max_cm[iD] << ", from_cm[iD] : " << from_cm[iD] << std::endl;
			//UE_LOG(LogCameraCalibrationEditor, Display, TEXT("iD : %d / %d, p_min_cm[iD] : %f, p_max_cm[iD] : %f, p_pad_cm[iD] : %f, p_interval_cm[iD] : %f, from_cm[iD] : %f"), iD, n_dim, p_min_cm[iD], p_max_cm[iD], p_pad_cm[iD], p_interval_cm[iD], from_cm[iD]);
		}
		std::vector<int> nn(n_dim);
		int n_obj = 1;
		for (int iD = 0; iD < n_dim; iD++)
		{
			double t_0 = p_max_cm[iD] + p_pad_cm[iD];
			int t_1 = floor((t_0 - from_cm[iD]) / p_interval_cm[iD]);
			int t_2 = t_1;
			while (double(t_2) * p_interval_cm[iD] + from_cm[iD] < t_0)
			{
				t_2++;
			}
			t_2++;
			nn[iD] = t_2;
			n_obj *= nn[iD];
			//UE_LOG(LogCameraCalibrationEditor, Display, TEXT("iD : %d / %d, t_0 : %f, t_1 : %d, t_2 : %d"), iD, n_dim, t_0, t_1, t_2);
		}
		//exit(0);
		std::vector<int> n_acc(n_dim);
		int iD = 0;
		for (; iD < n_dim - 1; iD++)
		{
			n_acc[iD] = 1;
			for (int i2 = n_dim - 1; i2 > iD; i2--)
			{
				n_acc[iD] *= nn[i2];
			}
			//std::cout << "iD : " << iD << " / " << n_dim << ", nn[iD] : " << nn[iD] << ", n_acc[iD] : " << n_acc[iD] << " / " << n_obj << ", n_acc[iD] * nn[iD] = " << n_acc[iD] * nn[iD] << std::endl;
			//UE_LOG(LogCameraCalibrationEditor, Display, TEXT("iD : %d / %d, nn[iD] : %d, n_acc[iD] : %d / %d, n_acc[iD] * nn[iD] = %ld"), iD, n_dim, nn[iD], n_acc[iD], n_obj, n_acc[iD] * nn[iD]);
		}
		n_acc[iD] = 1;
		std::cout << "iD : " << iD << " / " << n_dim << ", nn[iD] : " << nn[iD] << ", n_acc[iD] : " << n_acc[iD] << " / " << n_obj << ", n_acc[iD] * nn[iD] = " << n_acc[iD] * nn[iD] << std::endl;
		//UE_LOG(LogCameraCalibrationEditor, Display, TEXT("iD : %d / %d, nn[iD] : %d, n_acc[iD] : %d / %d, n_acc[iD] * nn[iD] = %ld"), iD, n_dim, nn[iD], n_acc[iD], n_obj, n_acc[iD] * nn[iD]);
		//std::vector<cv::Mat> t0;
		//UE_LOG(LogCameraCalibrationEditor, Display, TEXT("t0.max_size() : %ld"), t0.max_size());

		//int t2 = 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2;
		//UE_LOG(LogCameraCalibrationEditor, Display, TEXT("t2 : %ld, n_obj : %ld"), t2, n_obj);
		//std::vector<cv::Mat> li_obj_cm(t2);
		std::vector<cv::Mat> li_obj_cm(n_obj);
		std::vector<std::vector<int> > li_li_idx;
		for (int iO = 0; iO < n_obj; iO++)
		{
			//std::cout << "iO : " << iO << " / " << n_obj << std::endl;
			std::vector<int> li_idx(n_dim);
			iD = 0;
			int remained = iO;
			for (; iD < n_dim - 1; iD++)
			{
				li_idx[iD] = floor(remained / n_acc[iD]);
				remained -= li_idx[iD] * n_acc[iD];
				//std::cout << "\tiD : " << iD << " / " << n_dim << ", li_idx[iD] : " << li_idx[iD] << " / " << nn[iD] << std::endl;
			}
			li_idx[iD] = remained;
			//std::cout << "\tiD : " << iD << " / " << n_dim << ", li_idx[iD] : " << li_idx[iD] << " / " << nn[iD] << std::endl;
			li_li_idx.push_back(li_idx);
		}
		for (int iO = 0; iO < n_obj; iO++)
		{
			//std::cout << "iO : " << iO << " / " << n_obj << std::endl;
			li_obj_cm[iO] = cv::Mat::zeros(n_dim, 1, CV_64FC1);
			for (iD = 0; iD < n_dim; iD++)
			{
				int idx = li_li_idx[iO][iD];
				//cout << "idx : " << idx << endl;
				double cm = from_cm[iD] + (double)idx * p_interval_cm[iD];
				//cout << "cm : " << cm << endl;
				li_obj_cm[iO].at<double>(iD, 0) = cm;
				//std::cout << "\tiD : " << iD << " / " << n_dim << ", idx : " << idx << " / " << nn[iD] << ", cm : " << cm << std::endl;
			}
		}
		return std::tuple<std::vector<int>, std::vector<int>, std::vector<cv::Mat> >(nn, n_acc, li_obj_cm);
	}


bool apply_tps(tuple<vector<cv::Mat>, vector<vector<double> >, vector<double>, vector<double> >& tu_li_mat_src_li_li_tgt_li_pad_li_interv)
{
	std::vector<cv::Mat> li_mat_src = std::get<0>(tu_li_mat_src_li_li_tgt_li_pad_li_interv);
	std::vector<vector<double> > li_li_tgt = std::get<1>(tu_li_mat_src_li_li_tgt_li_pad_li_interv);
	vector<double> p_padding = std::get<2>(tu_li_mat_src_li_li_tgt_li_pad_li_interv);
	vector<double> p_interval = std::get<3>(tu_li_mat_src_li_li_tgt_li_pad_li_interv);
	cv::Mat param;
	bool is_param_computed = compute_parameters(param, li_mat_src, li_li_tgt);
	if(is_param_computed)
	{
		std::pair<std::vector<double>, std::vector<double> > pa_min_max = find_input_volume(li_mat_src);

		std::tuple<std::vector<int>, std::vector<int>, std::vector<cv::Mat> > tu_n_bin_n_acc_li_obj_to_warp = generate_list_of_obj_to_warp(pa_min_max, p_padding, p_interval);



		std::vector<int> n_bin = std::get<0>(tu_n_bin_n_acc_li_obj_to_warp);
		std::vector<int> n_acc = std::get<1>(tu_n_bin_n_acc_li_obj_to_warp);
		std::vector<cv::Mat> li_obj_to_warp_cm = std::get<2>(tu_n_bin_n_acc_li_obj_to_warp);
		int n_dim = p_interval.size(), num_points = li_mat_src.size(), num_points_obj = li_obj_to_warp_cm.size();
		//UE_LOG(LogCameraCalibrationEditor, Display, TEXT("num_points : %d, num_points_obj : %d"), num_points, num_points_obj);
		cv::Mat K2 = cv::Mat::zeros(num_points_obj, num_points, CV_64FC1);
		for (int iN = 0; iN < num_points_obj; iN++)
		{
			for (int iO = 0; iO < num_points; iO++)
			{
				double dist = MAX(0.00000001, norm(li_mat_src[iO] - li_obj_to_warp_cm[iN]));
				K2.at<double>(iN, iO) = dist;
			}
		}
#ifdef KEVIN_DEBUG
		t_k2 = std::chrono::steady_clock::now();
		UE_LOG(LogCameraCalibrationEditor, Display, TEXT("Lap for k2 : %d [secs]"), std::chrono::duration_cast<std::chrono::seconds>(t_k2 - t_gen).count());
#endif	//	KEVIN_DEBUG
		cv::Mat P2 = cv::Mat::ones(num_points_obj, n_dim + 1, CV_64FC1);
		for (int iN = 0; iN < num_points_obj; iN++)
		{
			for (int iD = 0; iD < n_dim; iD++)
			{
				P2.at<double>(iN, iD + 1) = li_obj_to_warp_cm[iN].at<double>(iD, 0);
			}
		}
#ifdef KEVIN_DEBUG
		t_p2 = std::chrono::steady_clock::now();
		UE_LOG(LogCameraCalibrationEditor, Display, TEXT("Lap for p2 : %d [secs]"), std::chrono::duration_cast<std::chrono::seconds>(t_p2 - t_k2).count());
#endif	//	KEVIN_DEBUG
		cv::Mat L2;  hconcat(K2, P2, L2);
#ifdef KEVIN_DEBUG
		t_concat = std::chrono::steady_clock::now();
		UE_LOG(LogCameraCalibrationEditor, Display, TEXT("Lap for concat : %d [secs]"), std::chrono::duration_cast<std::chrono::seconds>(t_concat - t_p2).count());
#endif	//	KEVIN_DEBUG
		cv::Mat object_warped = L2 * param;
		cout << "object_warped : " << endl << object_warped << endl;
	}


	return true;
}

int main(int argc, char **argv)
{
	if(argc < 2) 
    {
        cout << "Usage : " << argv[0] << "[# dimension]" << endl;	return 0;
    }
    if(!is_only_number(argv[1]))
    {
        cout << "The given input " << argv[1] << " is NOT a number" << endl;	return 0;
    }
    srand(time(NULL));
    int n_dim = atoi(argv[1]);

    tuple<vector<cv::Mat>, vector<vector<double> >, vector<double>, vector<double> > tu_li_mat_src_li_li_tgt_li_pad_li_interv = generate_src_tgt(n_dim);
    apply_tps(tu_li_mat_src_li_li_tgt_li_pad_li_interv);

	return 0;
}
