package com.example.timofeev;

import androidx.core.util.Pair;

import java.util.Vector;

public class Cluster {
    private final Vector<Pair<Integer, float[]>> centers = new Vector<>();
    private final Vector<Integer> counters = new Vector<>();
    private final float threashold = 0.7f;

    public int add(float[] v){
        Vector<Float> dists = new Vector<>();
        float min_dist = 2.0f;
        int min_idx = -1;
        float[] norm_vec_ = norm_vec(v);
        for(int i=0;i<centers.size();i++){
            if(dist(norm_vec_, centers.get(i).second) < min_dist) {
                min_dist = dist(norm_vec_, centers.get(i).second);
                min_idx = i;
            }
        }
        if(min_idx == -1){
            counters.add(1);
            centers.add(new Pair<Integer, float[]>(0, norm_vec_));
            return 0;
        } else {
            if(min_dist < threashold){
                int counter = counters.get(min_idx);
                int id = centers.get(min_idx).first;
                float[] new_center = new float[v.length];
                float[] center = centers.get(min_idx).second;
                for(int i=0;i<center.length;i++)
                    new_center[i] = (center[i] * counter + norm_vec_[i]) / (counter + 1);
                counters.set(min_idx, counter + 1);
                centers.set(min_idx, new Pair<Integer, float[]>(id, new_center));
                return id;
            } else {
                counters.add(1);
                centers.add(new Pair<Integer, float[]>(centers.size(), norm_vec_));
                return 0;
            }
        }
    }

    public int get(float[] v){
        Vector<Float> dists = new Vector<>();
        float min_dist = 2.0f;
        int min_idx = -1;
        float[] norm_vec_ = norm_vec(v);
        for(int i=0;i<centers.size();i++){
            if(dist(norm_vec_, centers.get(i).second) < min_dist) {
                min_dist = dist(norm_vec_, centers.get(i).second);
                min_idx = i;
            }
        }
        if(min_idx == -1){
            return -1;
        }
        return centers.get(min_idx).first;
    }

    private float dist(float[] v1, float[] v2){
        float sum = 0;
        for(int i=0;i<v1.length;i++)
            sum += v1[i] * v2[i];
        return 1 - sum;
    }

    private float norm(float[] v){
        float s = 0.0f;
        for(float e: v)
            s += e * e;
        return (float)Math.sqrt(s);
    }

    private float[] norm_vec(float[] v){
        float norm_ = norm(v);
        float[] ret = new float[v.length];
        for(int i=0;i<v.length;i++)
            ret[i] = v[i] / norm_;
        return ret;
    }
}
