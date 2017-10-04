//
// Created by fotoable on 2017/10/4.
//

#ifndef NCNN_NORMALIZEDBBOX_H
#define NCNN_NORMALIZEDBBOX_H


class NormalizedBBox {
public:
    NormalizedBBox(){};
    float xmin()const {return xmin_;}
    float ymin()const {return ymin_;}
    float xmax()const {return xmax_;}
    float ymax()const { return  ymax_;}
    int label()const{ return label_;}
    bool difficult()const{return difficult_;}
    float score()const{return score_;}
    float size()const{return size_;}

    void set_xmin(float val){xmin_ = val; has_xmin_ = true;}
    void set_ymin(float val){ymin_ = val; has_ymin_ = true;}
    void set_xmax(float val){xmax_ = val; has_xmax_ = true;}
    void set_ymax(float val){ymax_ = val; has_ymax_ = true;}
    void set_label(int val){label_ = val;has_label_ = true;}
    void set_difficult(bool val){difficult_=val;has_difficult_ = true;}
    void set_score(float val){score_=val;has_score_ = true;}
    void set_size(float val){size_ = val;has_size_=true;}
    bool has_size()const{return has_size_;}
    void clear_size(){size_=0;has_size_ = false;}

private:
    bool has_xmin_ = false;
    bool has_ymin_ = false;
    bool has_xmax_ = false;
    bool has_ymax_ = false;
    bool has_label_ = false;
    bool has_difficult_ = false;
    bool has_score_ = false;
    bool has_size_ = false;
    float xmin_;
    float ymin_;
    float xmax_;
    float ymax_;
    int label_;
    bool difficult_;
    float score_;
    float size_;


};


#endif //NCNN_NORMALIZEDBBOX_H
