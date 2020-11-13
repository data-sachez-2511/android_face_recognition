package com.example.timofeev;

import android.graphics.Bitmap;

import org.opencv.core.Mat;

import java.util.Vector;

public class Pipe {
    private String detector_path;
    private String embedder_path;
    private Detector detector;
    private Embedder embedder;

    public Pipe(String detector_path_, String embedder_path_){
        detector_path = detector_path_;
        embedder_path = embedder_path_;
        detector = new Detector(detector_path);
        embedder = new Embedder(embedder_path);
    }

    public Vector<float[]> pipe(Bitmap bitmap){
        Vector<Mat> faces = detector.detect(bitmap);
        Vector<float[]> embeds = new Vector<>();
        for(Mat face: faces)
            embeds.add(embedder.forward(face));
        return embeds;
    }
}
