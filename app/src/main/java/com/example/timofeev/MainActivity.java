package com.example.timofeev;


import android.app.FragmentManager;
import android.app.FragmentTransaction;
import android.content.Context;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;

import android.view.View;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;


import org.pytorch.Module;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;

/**
 * Created by avsavchenko.
 */

public class MainActivity extends FragmentActivity {

    /** Tag for the {@link Log}. */
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;

    private ProgressBar progressBar;
    private TextView progressBarinsideText;

    private Thread photoProcessingThread=null;
    private Map<String,Long> photosTaken;
    private ArrayList<String> photosFilenames;
    private int currentPhotoIndex=0;
    private Pipe pipe;
    private Cluster cluster = new Cluster();

    private String[] categoryList;
    private HashMap<Integer, Vector<Integer>> idx_map = new HashMap<>();

    private List<Map<String,Map<String, Set<String>>>> categoriesHistograms=new ArrayList<>();
    private List<Map<String, Map<String, Set<String>>>> eventTimePeriod2Files=new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        String detector_path = "";
        String embedder_path = "";
        try {
            for (String asset : getAssets().list("")){
                if(asset.toLowerCase().endsWith(".pt")){
                    if(asset.contains("detector")){
                        detector_path = assetFilePath(this, asset);
                    }
                    if(asset.contains("embedder")){
                        embedder_path = assetFilePath(this, asset);
                    }
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Error reading assets: " + e+" "+Log.getStackTraceString(e));
        }
        pipe = new Pipe(detector_path, embedder_path);
        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
        }
        else
            init();
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private void addDayEvent(List<Map<String, Map<String, Set<String>>>> eventTimePeriod2Files, String category, String timePeriod, Set<String> filenames){
        int highLevelCategory=getHighLevelCategory(category);
        if(highLevelCategory>=0) {
            Map<String,Map<String,Set<String>>> histo=eventTimePeriod2Files.get(highLevelCategory);
            if(!histo.containsKey(category))
                histo.put(category,new TreeMap<>(Collections.reverseOrder()));
            histo.get(category).put(timePeriod,filenames);

            //Log.d(TAG,"EVENTS!!! "+timePeriod+":"+category+" ("+highLevelCategory+"), "+filenames.size());
        }
    }

    private void init(){
        //checkServerSettings();
        photosTaken = photoProcessor.getCameraImages();
        photosFilenames=new ArrayList<String>(photosTaken.keySet());
        currentPhotoIndex=0;

        progressBar=(ProgressBar) findViewById(R.id.progress);
        progressBar.setMax(photosFilenames.size());
        progressBarinsideText=(TextView)findViewById(R.id.progressBarinsideText);
        progressBarinsideText.setText("");


        photoProcessingThread = new Thread(){
            public void run(){
                processAllPhotos();
            }
        };
        progressBar.setVisibility(View.VISIBLE);

        photoProcessingThread.setPriority(Thread.MIN_PRIORITY);
        photoProcessingThread.start();
    }
    public synchronized List<Map<String,Map<String, Set<String>>>> getCategoriesHistograms(boolean allLogs){
        if (allLogs)
            return categoriesHistograms;
        else
            return eventTimePeriod2Files;
    }

    private void processAllPhotos(){
        //ImageAnalysisResults previousPhotoProcessedResult=null;
        for(;currentPhotoIndex<photosTaken.size();++currentPhotoIndex){
            String filename=photosFilenames.get(currentPhotoIndex);
            try {
                File file = new File(filename);

                if (file.exists()) {
                    long startTime = SystemClock.uptimeMillis();
                    Bitmap myBitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
                    Vector<float[]> embeds = new Vector<>();
                    for(float[] embed: embeds){
                        int id = cluster.add(embed);
                        if(idx_map.containsKey(currentPhotoIndex)){
                            Objects.requireNonNull(idx_map.get(currentPhotoIndex)).add(id);
                        } else {
                            idx_map.put(currentPhotoIndex, new Vector<Integer>());
                        }
                    }

                    long endTime = SystemClock.uptimeMillis();
                    Log.d(TAG, "!!Processed: "+ filename+" in background thread:" + Long.toString(endTime - startTime));
                    final int progress=currentPhotoIndex+1;
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (progressBar != null) {
                                progressBar.setProgress(progress);
                                progressBarinsideText.setText("" + 100 * progress / photosTaken.size() + "%");
                            }
                        }
                    });
                }
            } catch (Exception e) {
                e.printStackTrace();
                Log.e(TAG, "While  processing image" + filename + " exception thrown: " + e);
            }
        }
    }

    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    getPackageManager()
                            .getPackageInfo(getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }
    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            int status=ContextCompat.checkSelfPermission(this,permission);
            if (ContextCompat.checkSelfPermission(this,permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS:
                Map<String, Integer> perms = new HashMap<String, Integer>();
                boolean allGranted = true;
                for (int i = 0; i < permissions.length; i++) {
                    perms.put(permissions[i], grantResults[i]);
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED)
                        allGranted = false;
                }
                // Check for ACCESS_FINE_LOCATION
                if (allGranted) {
                    // All Permissions Granted
                    init();
                } else {
                    // Permission Denied
                    Toast.makeText(MainActivity.this, "Some Permission is Denied", Toast.LENGTH_SHORT)
                            .show();
                    finish();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }
}
