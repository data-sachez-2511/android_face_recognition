package com.example.timofeev;

//import Math.exp;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;

import androidx.annotation.RequiresApi;

import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class Detector {
    private String model_path;
    private final String tag = "Detector";
    private final int model_size = 256;
    private Module model;

    private final Vector<float[]> prior_box_ = new Vector<>();
    private final float[] variance = {0.1f, 0.2f};
    private final float confidence_threshold = 0.5f;
    private final float nms_threshold = 0.6f;


    public static float[] mean = new float[]{104.0f / 255.0f, 117.0f/ 255.0f , 123.0f/ 255.0f };
    public static float[] std = new float[]{1.0f, 1.0f, 1.0f};
    private float scale_ = 1.0f;

    Detector(String model_path_) {
        model_path = model_path_;
        model = Module.load(model_path);
        init_prior_data();
//        ArrayList<Float> string_shape;
//        int count = 0;
//        for(float[] b: prior_box_){
//            string_shape = new ArrayList<>();
//            for(float a: b)
//                string_shape.add(a);
//            Log.i(tag, "loc[" + count + "]: " + string_shape.toString());
//            count ++;
//        }
    }

    private Mat preprocessing(Mat m) {

//        Log.i(tag, "input image size: " + m.toString());
//        Log.i(tag, "input image: " + m.toString());
        int h = m.rows();
        int w = m.cols();
        int c = m.channels();
        scale_ = 1.0f;
        if (h > w && h > model_size) {
            scale_ = (float) model_size / h;
        }
        if (h <= w && w > model_size) {
            scale_ = (float) model_size / w;
        }
        if (scale_ < 1.0f) {
            h = (int) (h * scale_);
            w = (int) (w * scale_);
            Imgproc.resize(m, m, new Size(w, h));
        }
//        Log.i(tag, "scale: " + String.valueOf(scale_));
//        Log.i(tag, "resized image size: " + m.size().toString());
        int numChannels = m.channels();
        int frameSize = m.rows() * m.cols();
        byte[] temp = new byte[numChannels];
//        Log.i(tag, "numChannels: " + String.valueOf(numChannels));
        Mat ret_mat = new Mat(model_size, model_size, CvType.CV_8UC3, Scalar.all(0));
        for (int i = 0; i < m.cols(); i++) {
            for (int j = 0; j < m.rows(); j++) {
                m.get(j, i, temp);
                ret_mat.put(j, i, temp);
//                Log.i(tag, "input " + i + ", " + j + ", " + 0 + ": " + temp[0]);
//                Log.i(tag, "input " + i + ", " + j + ", " + 0 + ": " + temp[1]);
//                Log.i(tag, "input " + i + ", " + j + ", " + 0 + ": " + temp[2]);
            }
        }
//        Log.i(tag, "preprocessing image size: " + ret_mat.size().toString());
        return ret_mat;
    }

    private Vector<float[]> postprocessing(float[] tensor) {
        Vector<float[]> loc = new Vector<>();
        Vector<Float> scores = new Vector<>();
        // из выхода модели можно получить точки лица, однако выравнивание лица не будет в проекте
        for (int i = 0; i < tensor.length; i += 16) {
            loc.add(Arrays.copyOfRange(tensor, i, i + 4));
            scores.add(tensor[i + 5]);
        }
//        Log.i(tag, "postprocessing loc size: " + loc.size());
//        Log.i(tag, "postprocessing scores size: " + scores.size());
//        Log.i(tag, "postprocessing scores max: " + Collections.max(scores));
        ArrayList<Float> string_shape = new ArrayList<>();
        for (float a : loc.get(0)) {
            string_shape.add(a);
        }
//        Log.i(tag, "loc[0]: " + string_shape.toString());
//        Log.i(tag, "scores: " + scores.toString());
        Vector<float[]> boxes = decode_batch(loc);
        string_shape = new ArrayList<>();
        for (float a : boxes.get(0)) {
            string_shape.add(a);
        }
//        Log.i(tag, "boxes[0]: " + string_shape.toString());
//        Log.i(tag, "postprocessing boxes size: " + boxes.size());

        Vector<Integer> idxs = new Vector<>();
        for(int i=0;i<scores.size();i++){
            if(scores.get(i) > confidence_threshold)
                idxs.add(i);
        }
//        Log.i(tag, "postprocessing idxs: " + idxs.toString());

        Vector<float[]> scored_dets = new Vector<>();
        for(int i=0;i<idxs.size();i++){
            float[] box =  boxes.get(idxs.get(i));
            float score = scores.get(idxs.get(i));
            float [] det = {box[0], box[1], box[2], box[3], score};
            scored_dets.add(det);
        }
        return nms(scored_dets);
    }

    private Vector<float[]> nms(final Vector<float[]> dets){
        //argsort
        Integer[] order = new Integer[dets.size()];
        for (int i = 0; i < dets.size(); i++) {
            order[i] = i;
        }
        Arrays.sort(order, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return Float.compare(dets.get(i2)[4], dets.get(i1)[4]);
            }
        });
        //nms
        LinkedList<Integer> idxs = new LinkedList<>(Arrays.asList(order));
        Vector<Integer> keep = new Vector<>();
        while (idxs.size() > 0){
            int m = idxs.get(0);
            float[] bm = dets.get(m);
            keep.add(m);
            idxs.remove(0);
            LinkedList<Integer> idxs_copy = new LinkedList<>();
            for(int idx: idxs){
                float[] b = dets.get(idx);
                float bx1 = Math.max(bm[0], b[0]);
                float by1 = Math.max(bm[1], b[1]);
                float bx2 = Math.max(bm[2], b[2]);
                float by2 = Math.max(bm[3], b[3]);
                float width = bx2 - bx1;
                float height = by2 - by1;
                float area_overlap = width * height;
                float area_a = (bm[2] - bm[0]) * (bm[3] - bm[1]);
                float area_b = (b[2] - b[0]) * (b[3] - b[1]);
                float area_combined = area_a + area_b - area_overlap;
                float iou = area_overlap / (area_combined+0.00000001f);
                if(iou < nms_threshold)
                    idxs_copy.add(idx);
            }
            idxs = new LinkedList<>(idxs_copy);
        }
        Vector<float[]> ret = new Vector<>();
        for(int idx: keep)
            ret.add(dets.get(idx));
        return ret;
    }

    private Vector<float[]> decode_batch(Vector<float[]> loc) {
        Vector<float[]> boxes = new Vector<>();
        for (int i = 0; i < loc.size(); i++) {
            float b1 = prior_box_.get(i)[0] + loc.get(i)[0] * variance[0] * prior_box_.get(i)[2];
            float b2 = prior_box_.get(i)[1] + loc.get(i)[1] * variance[0] * prior_box_.get(i)[3];
            float b3 = prior_box_.get(i)[2] * (float) Math.exp(loc.get(i)[2] * variance[1]);
            float b4 = prior_box_.get(i)[3] * (float) Math.exp(loc.get(i)[3] * variance[1]);
//            Log.i(tag, "box[" + i + "]: " + b1 + ", " + b2 + ", " + b3 + ", " + b4);
            b1 -= b3 / 2;
            b2 -= b4 / 2;
            b3 += b1;
            b4 += b2;
            boxes.add(new float[]{b1 * model_size / scale_, b2 * model_size / scale_, b3 * model_size / scale_, b4 * model_size / scale_});
        }
        return boxes;
    }

    private void init_prior_data() {
        int[][] feauture_map = {{32, 32}, {16, 16}, {8, 8}};
        int[][] min_sizes = {{16, 32}, {64, 128}, {256, 512}};
        int[] steps = {8, 16, 32};
        for (int f_idx = 0; f_idx < 3; f_idx++) {
            int[] min_size = min_sizes[f_idx];
            for (int i1 = 0; i1 < feauture_map[f_idx][0]; i1++) {
                for (int i2 = 0; i2 < feauture_map[f_idx][1]; i2++) {
                    for (int size_idx = 0; size_idx < 2; size_idx++) {
                        float s_kx = (float) min_size[size_idx] / model_size;
                        float s_ky = (float) min_size[size_idx] / model_size;
                        float dense_cx = ((float) i2 + 0.5f) * steps[f_idx] / model_size;
                        float dense_cy = ((float) i1 + 0.5f) * steps[f_idx] / model_size;
                        float[] anchor = {dense_cx, dense_cy, s_kx, s_ky};
                        prior_box_.add(anchor);
                    }
                }
            }
        }
    }

    public Vector<Mat> detect(Bitmap bitmap){
        Mat m = new Mat(300, 300, CvType.CV_8UC3);
        Utils.bitmapToMat(bitmap, m);
        Imgproc.cvtColor(m, m, Imgproc.COLOR_RGBA2BGR);
        Mat copy_m = new Mat();
        m.copyTo(copy_m);
        Mat input = preprocessing(m);
        Tensor input_tensor = TensorUtils.to_tensor(input, mean, std);
        Tensor output_tensor = model.forward(IValue.from(input_tensor)).toTensor();
        float[] output = output_tensor.getDataAsFloatArray();
        Vector<float[]> dets = postprocessing(output);
        Vector<Mat> faces = new Vector<>();
        for(float[] det: dets)
            faces.add(get_face(copy_m, det));
        return faces;
    }

    private Mat get_face(Mat image, float[] det){
//        Mat face = new Mat();
        int x1 = (int)det[0];
        int y1 = (int)det[1];
        int x2 = (int)det[2];
        int y2 = (int)det[3];
//        Log.i(tag, "main image: " + image.toString());
        Rect rectCrop = new Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
//        Log.i(tag, "rectCrop: " + rectCrop.toString());

        Mat face = image.submat(rectCrop);
        Imgproc.cvtColor(face, face, Imgproc.COLOR_BGR2RGB);
        return face;
    }



}
