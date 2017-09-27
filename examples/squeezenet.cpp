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
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "net.h"
#include <iostream>
#include <cvaux.h>
#include "style.id.h"
void detect_squeezenet(const cv::Mat& bgr, const unsigned char* model_mem,const unsigned char* bin_mem,cv::Mat& result_mat)
{
    cv::Mat resized_bgr;
    cv::resize(bgr,resized_bgr,cv::Size(256,256));
    ncnn::Net squeezenet;
    //squeezenet.load_param_bin("style.param.bin");
    squeezenet.load_param(model_mem);
    //squeezenet.load_model("style.bin");
    squeezenet.load_model(bin_mem);

    ncnn::Mat in = ncnn::Mat::from_pixels(resized_bgr.data, ncnn::Mat::PIXEL_BGR,resized_bgr.cols,resized_bgr.rows);

    const float mean_vals[3] = {103.939f, 116.779f, 123.68f};
    const float minus_mean_vals[3] = {-104.f, -117.f, -123.f};
    const float zero_mean_vals[3] = {0, 0, 0};
    const float const_factors[3] = {150.0,150.0,150.0};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);

    ex.input(style_param_id::BLOB_data, in);

    ncnn::Mat out;
    ex.extract(style_param_id::BLOB_output, out);

    using namespace std;
    cout<<"w"<<out.w<<"h"<<out.h<<"c"<<out.c<<endl;
    unsigned char result[512*512*3];
    out.substract_mean_normalize(zero_mean_vals,const_factors);
    out.substract_mean_normalize(minus_mean_vals,0);
    out.to_pixels(result,ncnn::Mat::PIXEL_BGR);
    int out_size = 512;
    result_mat.data = result;
    cv::Mat outmat(out_size,out_size,CV_8UC3,result);
    cv::imwrite("result3.jpg",outmat);
    //cv::imshow("1",outmat);
}
cv::Mat ncnnStyleTransfer(cv::Mat img,const unsigned char*param,const unsigned char* bin,const float max_size,const int pad_size = 8)
{
    cv::Mat resized_img;
    int new_height = 0;
    int new_width = 0;
    if(img.size().height>img.size().width)
    {
         new_height = max_size;
         new_width = (max_size/img.size().height)*img.size().width;
    }
    else
    {
        new_width = max_size;
        new_height = (max_size/img.size().width)*img.size().height;

    }
    cv::resize(img,resized_img,cv::Size(new_width,new_height));
    if(resized_img.channels()==4)
    {
        cv::cvtColor(resized_img,resized_img,cv::COLOR_RGBA2BGR);
    }
    cv::Mat padded_img;
    cv::copyMakeBorder(resized_img,padded_img,pad_size,pad_size,pad_size,pad_size,cv::BORDER_REFLECT);
    ncnn::Net style_net;
    style_net.load_param(param);
    style_net.load_model(bin);
    ncnn::Mat in = ncnn::Mat::from_pixels(padded_img.data, ncnn::Mat::PIXEL_BGR,padded_img.cols,padded_img.rows);

    const float mean_vals[3] = {103.939f, 116.779f, 123.68f};
    const float minus_mean_vals[3] = {-104.f, -117.f, -123.f};
    const float zero_mean_vals[3] = {0, 0, 0};
    const float const_factors[3] = {150.0,150.0,150.0};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = style_net.create_extractor();
    ex.set_light_mode(true);

    ex.input(style_param_id::BLOB_data, in);

    ncnn::Mat out;
    ex.extract(style_param_id::BLOB_output, out);

    using namespace std;
    unsigned char result[out.w*out.h*out.c];
    out.substract_mean_normalize(zero_mean_vals,const_factors);
    out.substract_mean_normalize(minus_mean_vals,0);
    out.to_pixels(result,ncnn::Mat::PIXEL_BGR);
    cv::Mat outmat(out.h,out.w,CV_8UC3,result);
    cv::Rect crop_rec(pad_size*2,pad_size*2,2*new_width,2*new_height);
    outmat = outmat(crop_rec);
    cv::imshow("result4",outmat);
    return outmat;


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
    std::ifstream model_file("style_nr.param.bin",std::ios::binary);
    model_file.seekg(0,std::ios::end);//将文件指针移至文件尾
    int nn = model_file.tellg()/sizeof(unsigned char);//按整形大小计算的文件长度
    unsigned char model_mem[nn];
    model_file.seekg(0);//将文件指针移至文件开始的位置
    model_file.read((char*)&model_mem,sizeof(unsigned char)*nn);//读取第一个数到xx
    model_file.close();
    std::string bin_root = "";

    std::ifstream bin_file("/Users/fotoable/workplace/caffe_style/style_models/mobile_model/ncnn/style0/near/a022_R132NRC3S2_1_s512_w6_b4_10000_n.bin",std::ios::binary);
    bin_file.seekg(0,std::ios::end);//将文件指针移至文件尾
    nn = bin_file.tellg()/sizeof(unsigned char);//按整形大小计算的文件长度
    unsigned char bin_mem[nn];
    bin_file.seekg(0);//将文件指针移至文件开始的位置
    bin_file.read((char*)&bin_mem,sizeof(unsigned char)*nn);//读取第一个数到xx
    bin_file.close();
    cv::Mat result_2(512,512,CV_8UC3);
    result_2 = ncnnStyleTransfer(m,model_mem,bin_mem,360);
    cv::imshow("result2",result_2);
    cv::imwrite("result2.jpg",result_2);
    cv::waitKey();


    return 0;
}

