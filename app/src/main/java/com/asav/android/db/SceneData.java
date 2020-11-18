package com.asav.android.db;

import java.io.Serializable;
import java.util.*;

/**
 * Created by avsavchenko.
 */

public class SceneData implements ClassifierResult, Serializable {
    public ImageClassificationData scenes;


    public SceneData() {
    }

    public SceneData(Vector<Integer> clusters) {
        this.scenes = new ImageClassificationData(clusters);
    }

    public String toString() {
        String res = "result: " + scenes;
        return res;
    }
}
