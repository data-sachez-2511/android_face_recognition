package com.asav.android;

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

public class Embedder {
    private String model_path;
    private final String tag = "Embedder";
    private final int model_size = 112;
    private Module model;


    public static float[] mean = new float[]{ 0.5f, 0.5f, 0.5f, };
    public static float[] std = new float[]{0.5f, 0.5f, 0.5f};

    Embedder(String model_path_) {
        model_path = model_path_;
        model = Module.load(model_path);
    }

    float[] forward(Mat face){
        Imgproc.resize(face, face, new Size(model_size, model_size));
//        Log.i(tag, "Input face size: " + face.size().toString());
        Tensor input = TensorUtils.to_tensor(face, mean, std);
        Tensor output_tensor = model.forward(IValue.from(input)).toTensor();
//        Log.i(tag, "Output tensor desc: " + output_tensor.toString());
        return output_tensor.getDataAsFloatArray();
    }
}
