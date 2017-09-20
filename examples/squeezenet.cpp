// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "net.h"
#include <iostream>
#include <cvaux.h>

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    cv::Mat resized_bgr;
    cv::resize(bgr,resized_bgr,cv::Size(256,256));
    ncnn::Net squeezenet;
    squeezenet.load_param("style.param");
    squeezenet.load_model("style.bin");
    cv::imshow("input",bgr);

    ncnn::Mat in = ncnn::Mat::from_pixels(resized_bgr.data, ncnn::Mat::PIXEL_BGR,resized_bgr.cols,resized_bgr.rows);

    const float mean_vals[3] = {103.939f, 116.779f, 123.68f};
    const float minus_mean_vals[3] = {-104.f, -117.f, -123.f};
    const float zero_mean_vals[3] = {0, 0, 0};
    const float const_factors[3] = {150.0,150.0,150.0};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);
    //ex.extract("score_interp", out);

    using namespace std;
    cout<<"w"<<out.w<<"h"<<out.h<<"c"<<out.c<<endl;
    unsigned char result[512*512*3];
    out.substract_mean_normalize(zero_mean_vals,const_factors);
    out.substract_mean_normalize(minus_mean_vals,0);
    out.to_pixels(result,ncnn::Mat::PIXEL_BGR);
    int out_size = 512;
    cv::Mat outmat(out_size,out_size,CV_8UC3,result);
    //std::cout<<outmat.row(0)<<std::endl;
    cv::imwrite("result1.png",outmat);
    cv::imshow("result",outmat);
    cv::waitKey();
    cls_scores.resize(out.c);
/*
    for (int j=0; j<out.c; j++)
    {
        const float* prob = out.data + out.cstep * j;
        cls_scores[j] = prob[0];
    }

 */

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    //const char* imagepath = argv[1];
    const char* imagepath = "ppj.jpg";

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);

    fprintf(stderr, "cv::imread %s \n", imagepath);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores);

    print_topk(cls_scores, 3);

    return 0;
}

