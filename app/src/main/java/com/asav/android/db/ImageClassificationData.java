package com.asav.android.db;

import java.io.Serializable;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.TreeMap;
import java.util.Vector;

/**
 * Created by avsavchenko.
 */

public class ImageClassificationData implements ClassifierResult,Serializable {
    public Vector<Integer> categories =null;

    public ImageClassificationData(Vector<Integer> categories_){
        categories = categories_;
    }

    public String toString(){
        StringBuilder str=new StringBuilder();
        for (int i = 0; i< categories.size(); ++i){
            str.append(String.format("person %s; ", categories.get(i)));
        }
        return str.toString();
    }
}
