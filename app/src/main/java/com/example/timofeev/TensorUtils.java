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
public class TensorUtils {
    static final String tag = "TensorUtils";

    public static Tensor to_tensor(Mat mat, float[] mean, float[] std){
        Log.i(tag, "input image size: " + mat.toString());
        byte[] return_buff = new byte[(int) (mat.total() *
                mat.channels())];
        mat.get(0, 0, return_buff);
        float[] tensor = new float[return_buff.length];
        ArrayList<Float> data = new ArrayList<>();
        for(int c=0, it=0;c<mat.channels();c++){
            for(int i=c;i<return_buff.length;i+=mat.channels(), it++){
                tensor[it] = return_buff[i] < 0 ? (256 + return_buff[i]): return_buff[i];
                tensor[it] = (tensor[it] / 255.0f - mean[c]) / std[c];
            }
        }
        return Tensor.fromBlob(tensor, new long[]{1, mat.channels(), mat.rows(), mat.cols()});
    }


}
