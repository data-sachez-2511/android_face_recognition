package com.asav.android;


import android.app.FragmentManager;
import android.app.FragmentTransaction;
import android.content.Context;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;

import android.view.View;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.asav.android.db.EXIFData;
import com.asav.android.db.ImageAnalysisResults;
import com.asav.android.db.RectFloat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by avsavchenko.
 */

public class MainActivity extends FragmentActivity {

    /** Tag for the {@link Log}. */
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;

    private HighLevelVisualPreferences preferencesFragment;
    private Photos photosFragment;

    private ProgressBar progressBar;
    private TextView progressBarinsideText;

    private Thread photoProcessingThread=null;
    private Map<String,Long> photosTaken;
    private ArrayList<String> photosFilenames;
    private int currentPhotoIndex=0;
    private PhotoProcessor photoProcessor = null;

    private String[] categoryList;

    private List<Map<String,Map<String, Set<String>>>> categoriesHistograms=new ArrayList<>();
    private List<Map<String, Map<String, Set<String>>>> eventTimePeriod2Files=new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
        }
        else
            init();
    }
    private void init(){
        //checkServerSettings();
        categoryList = getResources().getStringArray(R.array.category_list);

        for(int i=0;i<categoryList.length-1;++i){
            categoriesHistograms.add(new HashMap<>());
        }

        for(int i=0;i<categoryList.length-2;++i){
            eventTimePeriod2Files.add(new HashMap<>());
        }

        photoProcessor = PhotoProcessor.getPhotoProcessor(this);
        photosTaken = photoProcessor.getCameraImages();
        photosFilenames=new ArrayList<String>(photosTaken.keySet());
        currentPhotoIndex=0;

        progressBar=(ProgressBar) findViewById(R.id.progress);
        progressBar.setMax(photosFilenames.size());
        progressBarinsideText=(TextView)findViewById(R.id.progressBarinsideText);
        progressBarinsideText.setText("");


        photoProcessingThread = new Thread(() -> {
            processAllPhotos();
        }, "photo-processing-thread");
        progressBar.setVisibility(View.VISIBLE);

        preferencesFragment = new HighLevelVisualPreferences();
        Bundle prefArgs = new Bundle();
        prefArgs.putInt("color", Color.GREEN);
        prefArgs.putString("title", "High-Level topCategories");
        preferencesFragment.setArguments(prefArgs);

        photosFragment=new Photos();
        Bundle args = new Bundle();
        args.putStringArray("photosTaken", new String[]{"0"});
        args.putStringArrayList("0",new ArrayList<String>(photoProcessor.getCameraImages().keySet()));
        photosFragment.setArguments(args);
        PreferencesClick(null);

        photoProcessingThread.setPriority(Thread.MIN_PRIORITY);
        photoProcessingThread.start();
    }
    public synchronized List<Map<String,Map<String, Set<String>>>> getCategoriesHistograms(boolean allLogs){
        Log.i(TAG, "ALLLOG: " + allLogs);
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
                    ImageAnalysisResults res = photoProcessor.getImageAnalysisResults(filename);

                    long endTime = SystemClock.uptimeMillis();
                    Log.d(TAG, "!!Processed: "+ filename+" in background thread:" + Long.toString(endTime - startTime));
                    processRecognitionResults(res);
                    final int progress=currentPhotoIndex+1;
                    runOnUiThread(() -> {
                        if(progressBar!=null) {
                            progressBar.setProgress(progress);
                            progressBarinsideText.setText(""+100*progress/photosTaken.size()+"%");
                        }
                    });
                }
            } catch (Exception e) {
                e.printStackTrace();
                Log.e(TAG, "While  processing image" + filename + " exception thrown: " + e);
            }
        }
    }



    private void updateCategory(List<Map<String,Map<String, Set<String>>>> histos, int highLevelCategory, String category, String filename){
        if(highLevelCategory>=0) {
            Map<String, Map<String, Set<String>>> histo = histos.get(highLevelCategory);
            Log.i(TAG, "QQQQQQQQQQQQ: " + histo);
            Log.i(TAG, "highLevelCategory: " + highLevelCategory);
            Log.i(TAG, "category: " + category);
            if (!histo.containsKey(category)) {
                histo.put(category, new TreeMap<>());
                histo.get(category).put(category, new TreeSet<>());
            }
            histo.get(category).get(category).add(filename);
        }
    }

    private List<Map<String,Map<String, Set<String>>>> deepCopyCategories(List<Map<String,Map<String, Set<String>>>> categories){
        ArrayList<Map<String,Map<String, Set<String>>>> result=new ArrayList<>(categories.size());
        for(Map<String,Map<String, Set<String>>> m:categories){
            Map<String,Map<String, Set<String>>> m1=new HashMap<>(m.size());
            result.add(m1);
            for(Map.Entry<String,Map<String, Set<String>>> me:m.entrySet()){
                Map<String, Set<String>> m2=new TreeMap<>(Collections.reverseOrder());
                m1.put(me.getKey(),m2);
                for(Map.Entry<String, Set<String>> map_files:me.getValue().entrySet()){
                    m2.put(map_files.getKey(),new TreeSet<>(map_files.getValue()));
                }
            }
        }
        return result;
    }
    private synchronized void processRecognitionResults(ImageAnalysisResults results){
        String filename=results.filename;
        if(!photoProcessor.file2cluster.containsKey(filename))
            return;

        List<Map<String,Map<String, Set<String>>>> newEventTimePeriod2Files = deepCopyCategories(eventTimePeriod2Files);
        photoProcessor.updateSceneInEvents(newEventTimePeriod2Files,filename);
        eventTimePeriod2Files=newEventTimePeriod2Files;
        //eventTimePeriod2Files=photoProcessor.updateSceneInEvents(categoryList.length-2);

        String location=results.locations.description;
        List<Map<String,Map<String, Set<String>>>> newCategoriesHistograms = deepCopyCategories(categoriesHistograms);

        String scenes = photoProcessor.file2cluster.get(filename);


        Log.i(TAG, "SCENES: " + scenes);
        updateCategory(newCategoriesHistograms, Integer.valueOf(scenes), scenes, filename);
//        for (String scene : scenes) {
//            updateCategory(newCategoriesHistograms, Integer.valueOf(scene), scene, filename);
//        }
//        if(location!=null)
//            updateCategory(newCategoriesHistograms, newCategoriesHistograms.size() - 1, location, filename);

        categoriesHistograms=newCategoriesHistograms;

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                preferencesFragment.updateChart();
            }
        });
    }

    public void PreferencesClick(View view) {
        FragmentManager fm = getFragmentManager();
        FragmentTransaction fragmentTransaction = fm.beginTransaction();
        fragmentTransaction.replace(R.id.fragment_switch, preferencesFragment);
        fragmentTransaction.commit();
    }
    public void PhotosClick(View view) {
        FragmentManager fm = getFragmentManager();
        if(fm.getBackStackEntryCount()==0) {
            FragmentTransaction fragmentTransaction = fm.beginTransaction();
            fragmentTransaction.replace(R.id.fragment_switch, photosFragment);
            fragmentTransaction.addToBackStack(null);
            fragmentTransaction.commit();
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

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    Log.i(TAG, "After loading all libraries" );
                    Toast.makeText(getApplicationContext(),
                            "OpenCV loaded successfully",
                            Toast.LENGTH_SHORT).show();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                    Toast.makeText(getApplicationContext(),
                            "OpenCV error",
                            Toast.LENGTH_SHORT).show();
                } break;
            }
        }
    };
}

