#include <algorithm>
#include "opencv2/opencv.hpp"
//#include "Timer.h"
#include <time.h>
#include <iterator>
#include <random>

using namespace std;
using namespace cv;

#if 0
std::string itos_formatted(int ii, int n_digit)
{
    std::stringstream ss;
    ss << std::setw(n_digit) << std::setfill('0') << ii;
    return ss.str();
}
#endif

bool is_only_number(const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(), 
	        s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}

tuple<vector<int>, vector<pair<vector<double>, vector<double> > >, 
    vector<pair<pair<double, double>, pair<double, double> > >,
    vector<int>, vector<double> > create_lut(int n_dim)
{
    //vector<double> p_min_cm = pa_min_max_cm.first, p_max_cm = pa_min_max_cm.second;
    std::vector<double> from(n_dim), li_interv(n_dim);
    std::vector<int> nn(n_dim);
    int n_obj = 1;
    vector<pair<pair<double, double>, pair<double, double> > > li_pa_pa_min_max_pa_min_max(n_dim);
    for (int iD = 0; iD < n_dim; iD++)
    {
        double interv = iD + 1;
        li_interv[iD] = interv;
        double min_v = (double)(iD);   
        double pad_1st = (double)(iD * iD); 
        from[iD] = min_v - pad_1st;      
        li_pa_pa_min_max_pa_min_max[iD] = pair<pair<double, double>, pair<double, double> >(
            pair<double, double>(from[iD], from[iD]), pair<double, double>(0, 0));
        int len = (iD + 1) * (iD + 2);             
        //double interv = (double)(iD + 1);
        nn[iD] = len;
        n_obj *= nn[iD];       
        std::cout << "iD : " << iD << " / " << n_dim << ", from[iD] : " << from[iD] << ", nn[iD] : " << nn[iD] << ", n_obj : " << n_obj << std::endl;
    }
    vector<int> li_n_acc(n_dim);
    int iD = 0;
    for (; iD < n_dim - 1; iD++)
    {
        li_n_acc[iD] = 1;
        for (int i2 = n_dim - 1; i2 > iD; i2--)
        {
            li_n_acc[iD] *= nn[i2];
        }
        std::cout << "iD : " << iD << " / " << n_dim << ", nn[iD] : " << nn[iD] << ", n_acc[iD] : " << li_n_acc[iD] << " / " << n_obj << ", n_acc[iD] * nn[iD] = " << li_n_acc[iD] * nn[iD] << std::endl;
    }
    li_n_acc[iD] = 1;
    std::cout << "iD : " << iD << " / " << n_dim << ", nn[iD] : " << nn[iD] << ", n_acc[iD] : " << li_n_acc[iD] << " / " << n_obj << ", n_acc[iD] * nn[iD] = " << li_n_acc[iD] * nn[iD] << std::endl;
    //exit(0);
    vector<pair<vector<double>, vector<double> > > li_pa_src_tgt(n_obj);
    vector<vector<int> > li_li_idx;
    for (int iO = 0; iO < n_obj; iO++)
    {
        //std::cout << "iO : " << iO << " / " << n_obj << std::endl;
        std::vector<int> li_idx(n_dim);
        iD = 0;
        int remained = iO;
        for (; iD < n_dim - 1; iD++)
        {
            li_idx[iD] = floor(remained / li_n_acc[iD]);
            remained -= li_idx[iD] * li_n_acc[iD];
            //std::cout << "\tiD : " << iD << " / " << n_dim << ", li_idx[iD] : " << li_idx[iD] << " / " << nn[iD] << std::endl;
        }
        li_idx[iD] = remained;
        //std::cout << "\tiD : " << iD << " / " << n_dim << ", li_idx[iD] : " << li_idx[iD] << " / " << nn[iD] << std::endl;
        li_li_idx.push_back(li_idx);
    }
    //exit(0);
    for (int iO = 0; iO < n_obj; iO++)
    {
        //std::cout << "iO : " << iO << " / " << n_obj << std::endl;
        vector<double> li_src(n_dim), li_tgt(n_dim);
        for (iD = 0; iD < n_dim; iD++)
        {
            int idx = li_li_idx[iO][iD];
            //cout << "\tidx : " << idx << endl;
            double interv = li_interv[iD];
            double v_src = from[iD] + (double)(idx * interv);
            double slope = iD + 1;
            double y_inter = -iD;
            double v_tgt = v_src * slope + y_inter;
            //cout << "cm : " << cm << endl;
            li_src[iD] = v_src;
            li_tgt[iD] = v_tgt;
            //std::cout << "\tiD : " << iD << " / " << n_dim << ", idx : " << idx << " / " << nn[iD] << ", v_src : " << v_src << ", v_tgt : " << v_tgt << std::endl;
            
            if(v_src > li_pa_pa_min_max_pa_min_max[iD].first.second)
            {
                //std::cout << "\tv_src : " << v_src << ", li_pa_min_max[iD].second b4 : " << li_pa_min_max[iD].second << std::endl;
                li_pa_pa_min_max_pa_min_max[iD].first.second = v_src;
                li_pa_pa_min_max_pa_min_max[iD].second.second = v_tgt;

                //std::cout << "\t\tli_pa_min_max[iD].second after : " << li_pa_min_max[iD].second << std::endl;
                //exit(0);
            }

            if(v_src <= li_pa_pa_min_max_pa_min_max[iD].first.first)
            {
                //std::cout << "\tv_src : " << v_src << ", li_pa_min_max[iD].second b4 : " << li_pa_min_max[iD].second << std::endl;
                li_pa_pa_min_max_pa_min_max[iD].first.first = v_src;
                li_pa_pa_min_max_pa_min_max[iD].second.first = v_tgt;
                //std::cout << "\t\tli_pa_min_max[iD].second after : " << li_pa_min_max[iD].second << std::endl;
                //exit(0);
            }
        }
        li_pa_src_tgt[iO] = pair<vector<double>, vector<double> >(li_src, li_tgt);
    }
    //exit(0);
    for (iD = 0; iD < n_dim; iD++)
    {
        std::cout << "iD : " << iD << ", li_pa_pa_min_max_pa_min_max[iD].first.first : " << li_pa_pa_min_max_pa_min_max[iD].first.first << ", li_pa_pa_min_max_pa_min_max[iD].first.second : " << li_pa_pa_min_max_pa_min_max[iD].first.second << std::endl;
    }
    //exit(0);
    return tuple<vector<int>, vector<pair<vector<double>, vector<double> > >, 
    vector< pair<pair<double, double>, pair<double, double> > >,
    vector<int>, vector<double> >(nn, li_pa_src_tgt, li_pa_pa_min_max_pa_min_max, li_n_acc, li_interv);  
}


vector<vector<int> > generate_permutations(int dimension)
{
	std::vector<std::vector<int> > result;
	if (0 == dimension)
	{
		result.push_back({});
	}
	else
	{
		std::vector<std::vector<int> > sub_permutations = generate_permutations(dimension - 1);
		for (const std::vector<int>& sub_permutation : sub_permutations)
		{
			for (int i = 0; i <= 1; i++)
			{
				std::vector<int> new_permutation = sub_permutation;
				new_permutation.push_back(i);
				result.push_back(new_permutation);
			}
		}
	}
	return result;
}


vector<double> interpolate_in_lut(const std::vector<double>& p_query, 
    const vector<pair<vector<double>, vector<double> > >& li_pa_src_tgt, 
    const vector<double>& li_interv, const vector<int>& li_n_bin, 
    const vector<vector<int> >& li_li_idx_naver, const vector<int>& li_n_acc)
{
#if 0
		std::cout << "p_query : (";
		for (int iD = 0; iD < n_dim; iD++) std::cout << p_query[iD] << ", ";
		std::cout << ")" << endl;   //exit(0);
#endif  //  0
    int n_dim = p_query.size();
    vector<double> p_000 = li_pa_src_tgt[0].first, p_interp;
	//vector<double> p_f(n_dim);
	std::vector<int> idx_0(n_dim);//, idx_1(n_dim);
	std::vector<double> r_delta(n_dim);
	for (int iD = 0; iD < n_dim; iD++)
	{
		double idx_f = (p_query[iD] - p_000[iD]) / li_interv[iD];
        idx_0[iD] = floor(idx_f);
		//idx_1[iD] = idx_0[iD] + 1;
		if (idx_0[iD] < 0 || idx_0[iD] + 1 >= li_n_bin[iD])
		{
		    std::cout << "idx_0[iD] : " << idx_0[iD] << ", li_n_bin[iD] : " << li_n_bin[iD] << std::endl;
			std::cout << iD << " th value of query, " << p_query[iD] << " is NOT in valid range of LUT, which is (" << p_000[iD] << " ~ " << p_000[iD] + double(li_n_bin[iD] - 1) * li_interv[iD] << ")" << std::endl;
			exit(0);
            return p_interp;
		}
		double v_0 = p_000[iD] + (double)idx_0[iD] * li_interv[iD];
		r_delta[iD] = (p_query[iD] - v_0) / li_interv[iD];
	}
#if 0
	std::vector<std::vector<int> > li_li_idx_naver = generate_permutations(n_dim);
	for (const std::vector<int>& combination : li_li_idx_naver)
	{
		for (int value : combination)
		{
			std::cout << value << " ";
		}
		std::cout << std::endl;
	}
#endif	//	0
	int n_naver = li_li_idx_naver.size();
	//std::vector<cv::Mat> li_interp_old(n_naver);
	vector<vector<double> > li_interp_old(n_naver);
	for (int iN = 0; iN < n_naver; iN++)
	{
		int idx_total = 0;
		for (int iD = 0; iD < n_dim; iD++)
		{
			int delta_i = li_li_idx_naver[iN][iD];
			int idx = idx_0[iD] + delta_i;
			idx_total += idx * li_n_acc[iD];
		}
		//li_interp_old[iN] = listed_table.row(idx_total);
		li_interp_old[iN] = li_pa_src_tgt[idx_total].second;
		//std::cout << "iN : " << iN << ", idx_total : " << idx_total << ", li_interp_old[iN] : " << li_interp_old[iN] << std::endl;
	}

	//exit(0);
#if 1   
	for (int iD = 0; iD < n_dim; iD++)
	{
		//std::cout << "iD : " << iD << " / " << n_dim << std::endl;
		n_naver /= 2;
		vector<vector<double> > li_interp_new(n_naver);
		for (int iN = 0; iN < n_naver; iN++)
		{
            li_interp_new[iN].resize(n_dim); 
            for(int iD2 = 0; iD2 < n_dim; iD2++)
            {
			    li_interp_new[iN][iD2] = (1.0 - r_delta[iD]) * li_interp_old[iN][iD2] + r_delta[iD] * li_interp_old[n_naver + iN][iD2];
    			//std::cout << "\tiN : " << iN << " / " << n_naver << ", r_delta[iD] : " << r_delta[iD] << ", li_interp_old[iN] : " << li_interp_old[iN] << ", li_interp_old[n_naver + iN] : " << li_interp_old[n_naver + iN] << ", li_interp_new[iN] : " << li_interp_new[iN] << std::endl;
		    }
        }
		li_interp_old = li_interp_new;
	}
#if 0
	for (int iD = 0; iD < n_dim; iD++)
	{
		//p_interp[iD] = li_interp_old[0].at<double>(0, iD);
	}
#endif  //  0    
	p_interp = li_interp_old[0];
#endif  //  0        
	return p_interp;
}

template<class T>
ostream& operator<<(ostream& stream, const std::vector<T>& values)
{
    copy( begin(values), end(values), ostream_iterator<T>(stream, ", ") );
        return stream;
        }

double uniform_rand()
{
    return (double)(rand()) / (double)(RAND_MAX);
}

int test_lut(tuple<
    vector<int>, 
    vector<pair<vector<double>, vector<double> > >, 
    vector<pair<pair<double, double>, pair<double, double> > >, 
    vector<int>, 
    vector<double>
>& tu_n_bin_li_pa_min_max_li_src_tgt_li_n_acc_li_interv)
{
    vector<int> li_n_bin = std::get<0>(tu_n_bin_li_pa_min_max_li_src_tgt_li_n_acc_li_interv);
    vector<pair<vector<double>, vector<double> > > li_pa_src_tgt = std::get<1>(tu_n_bin_li_pa_min_max_li_src_tgt_li_n_acc_li_interv);
    vector<pair<pair<double, double>, pair<double, double> > > li_pa_pa_min_max_pa_min_max = std::get<2>(tu_n_bin_li_pa_min_max_li_src_tgt_li_n_acc_li_interv);
    vector<int> li_n_acc = std::get<3>(tu_n_bin_li_pa_min_max_li_src_tgt_li_n_acc_li_interv);
    vector<double> li_interv = std::get<4>(tu_n_bin_li_pa_min_max_li_src_tgt_li_n_acc_li_interv);
    int n_dim = li_n_bin.size();
    vector<double> li_query(n_dim);
    for(int iD = 0; iD < n_dim; iD++)
    { 
        pair<double, double> pa_min_max = li_pa_pa_min_max_pa_min_max[iD].first;
        double from = pa_min_max.first, to = pa_min_max.second, ur = uniform_rand();
        li_query[iD] = from + ur * (to - from);           
        cout << "from : " << from << ", to : " << to << ", ur : " << ur << endl;
    }
    //exit(0);
	std::vector<std::vector<int> > li_li_idx_naver = generate_permutations(n_dim);
    vector<double> li_interp = interpolate_in_lut(li_query, li_pa_src_tgt, li_interv, li_n_bin, li_li_idx_naver, li_n_acc);
    for(int iD = 0; iD < n_dim; iD++)
    {
        pair<double, double> pa_min_max_src = li_pa_pa_min_max_pa_min_max[iD].first;
        pair<double, double> pa_min_max_tgt = li_pa_pa_min_max_pa_min_max[iD].second;
        cout << "iD : " << iD << ", li_query : " << pa_min_max_src.first << " / " << li_query[iD] << " / " << pa_min_max_src.second << ", li_interp : " << pa_min_max_tgt.first << " / " << li_interp[iD] << " / " << pa_min_max_tgt.second << endl;        
    }
    return 1;
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
