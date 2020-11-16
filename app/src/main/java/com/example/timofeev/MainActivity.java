package com.example.timofeev;


import android.annotation.SuppressLint;
import android.app.FragmentManager;
import android.app.FragmentTransaction;
import android.content.Context;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;

import android.provider.MediaStore;
import android.view.View;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.pytorch.Module;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;


public class MainActivity extends FragmentActivity {

    /** Tag for the {@link Log}. */
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;

    private ProgressBar progressBar;
    private TextView progressBarinsideText;

    private Thread photoProcessingThread=null;
    private ArrayList<String> photosFilenames;
    private HashMap<String, Integer> imageClusters;
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

//    private void addDayEvent(List<Map<String, Map<String, Set<String>>>> eventTimePeriod2Files, String category, String timePeriod, Set<String> filenames){
//        int highLevelCategory=getHighLevelCategory(category);
//        if(highLevelCategory>=0) {
//            Map<String,Map<String,Set<String>>> histo=eventTimePeriod2Files.get(highLevelCategory);
//            if(!histo.containsKey(category))
//                histo.put(category,new TreeMap<>(Collections.reverseOrder()));
//            histo.get(category).put(timePeriod,filenames);
//
//            //Log.d(TAG,"EVENTS!!! "+timePeriod+":"+category+" ("+highLevelCategory+"), "+filenames.size());
//        }
//    }

    private void init(){
        //checkServerSettings();
        photosFilenames = (ArrayList<String>) getImagePaths(this);
        imageClusters = new HashMap<>();
//        photosFilenames=new ArrayList<String>(photosTaken.keySet());
        Log.i(TAG, "photos: " + photosFilenames.toString());
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

    private void processAllPhotos() {
        //ImageAnalysisResults previousPhotoProcessedResult=null;
        for (; currentPhotoIndex < photosFilenames.size(); ++currentPhotoIndex) {
            String filename = photosFilenames.get(currentPhotoIndex);
            try {
                File file = new File(filename);

                if (file.exists()) {
                    long startTime = SystemClock.uptimeMillis();
                    Bitmap myBitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
                    Vector<float[]> embeds = pipe.pipe(myBitmap);
                    for (float[] embed : embeds) {
                        int id = cluster.add(embed);
                        imageClusters.put(filename, id);
                        Log.i(TAG, "CLUSTERS" + imageClusters.toString());
                        if (idx_map.containsKey(currentPhotoIndex)) {
                            Objects.requireNonNull(idx_map.get(currentPhotoIndex)).add(id);
                        } else {
                            idx_map.put(currentPhotoIndex, new Vector<Integer>());
                        }
                    }

                    long endTime = SystemClock.uptimeMillis();
                    Log.d(TAG, "!!Processed: " + filename + " in background thread:" + Long.toString(endTime - startTime));
                    final int progress = currentPhotoIndex + 1;
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (progressBar != null) {
                                progressBar.setProgress(progress);
                                progressBarinsideText.setText("" + 100 * progress / photosFilenames.size() + "%");
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

    public List<String> getImagePaths(Context context) {
        final String[] projection = {MediaStore.Images.Media.DATA, MediaStore.Images.Media.DATE_TAKEN};
        //String path= Environment.getExternalStorageDirectory().toString();//+"/DCIM/Camera";
        final String selection = null;//MediaStore.Images.Media.BUCKET_ID +" = ?";
        final String[] selectionArgs = null;//{String.valueOf(path.toLowerCase().hashCode())};
        ArrayList<String> result = new ArrayList<>();
        try {
            @SuppressLint("Recycle") final Cursor cursor = context.getContentResolver().query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, projection, selection, selectionArgs, MediaStore.Images.ImageColumns.DATE_TAKEN + " DESC");
            if (cursor.moveToFirst()) {
                int dataColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
                int dateColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_TAKEN);
                do {
                    String data = cursor.getString(dataColumn);
                    result.add(data);
                }
                while (cursor.moveToNext());
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "Exception thrown: " + e);
        }
        return result;
    }
}
