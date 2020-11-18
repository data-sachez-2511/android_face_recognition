package com.asav.android;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.location.Address;
import android.location.Geocoder;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import com.asav.android.db.ImageAnalysisResults;
import com.asav.android.db.EXIFData;
import com.asav.android.db.RectFloat;
import com.asav.android.db.SceneData;

import java.io.EOFException;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.text.DateFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by avsavchenko.
 */
public class PhotoProcessor {
    /**
     * Tag for the {@link Log}.
     */
    private static final String TAG = "PhotoProcessor";

    private Pipe pipe;
    private Cluster cluster = new Cluster();

    private ConcurrentHashMap<String, SceneData> scenes = new ConcurrentHashMap<>();
    private static final String IMAGE_SCENES_FILENAME = "image_scenes";

    private ConcurrentHashMap<String, EXIFData> exifs = new ConcurrentHashMap<>();
    private static final String IMAGE_EXIF_FILENAME = "image_exif";

    private static final boolean resetScenesModel = false;

    public final HashMap<String, String> file2cluster = new HashMap<>();
    public final HashMap<String, Set<String>> cluster2files = new HashMap<>();


    private final Context context;
    private Geocoder geocoder;

    private Map<String, Long> photosTaken = new LinkedHashMap<>();
    private Map<String, Set<String>> date2files = new LinkedHashMap<>();
    private static final int MIN_PHOTOS_PER_DAY = 3;
    private int avgNumPhotosPerDay = MIN_PHOTOS_PER_DAY;


    private Map<String, SceneData> file2Scene = new LinkedHashMap<>();
    private List<Map<String, Long>> sceneClusters = new LinkedList<>();


    private PhotoProcessor(final Activity context) {
        this.context = context;
        geocoder = new Geocoder(context, Locale.US);//, Locale.getDefault());
        initPhotosTaken();
        loadImageResults();
        loadModels();
    }

    private static PhotoProcessor instance;

    public static PhotoProcessor getPhotoProcessor(final Activity context) {
        if (instance == null) {
            instance = new PhotoProcessor(context);
        }
        return instance;
    }

    public Map<String, SceneData> getFile2Scene() {
        return file2Scene;
    }


    private void loadModels() {
        String detector_path = "";
        String embedder_path = "";
        try {
            for (String asset : context.getAssets().list("")){
                if(asset.toLowerCase().endsWith(".pt")){
                    if(asset.contains("detector")){
                        detector_path = assetFilePath(context, asset);
                    }
                    if(asset.contains("embedder")){
                        embedder_path = assetFilePath(context, asset);
                    }
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Error reading assets: " + e+" "+Log.getStackTraceString(e));
        }
        pipe = new Pipe(detector_path, embedder_path);
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

    private static <V> ConcurrentHashMap<String, V> readObjectMap(Context context, String filename) {
        long startTime = SystemClock.uptimeMillis();
        ConcurrentHashMap<String, V> map = new ConcurrentHashMap<String, V>();
        try {
            ObjectInputStream is = new ObjectInputStream(context.openFileInput(filename));
            try {
                String key;
                key = (String) is.readObject();
                while ((key = (String) is.readObject()) != null) {
                    V val = (V) is.readObject();
                    map.put(key, val);
                }
            } catch (EOFException eofEx) {
                Log.e(TAG, "EOF loading image results " + filename + " current size=" + map.size() + " exception thrown: " + eofEx);
            }
            is.close();
        } catch (Exception e) {
            Log.e(TAG, "While loading image results " + filename + " exception thrown: ", e);
        }
        Log.w(TAG,"Size of "+filename+" is "+context.getFileStreamPath(filename).length()+" Timecost: " + Long.toString(SystemClock.uptimeMillis() - startTime));
        return map;
    }

    private static class AppendingObjectOutputStream extends ObjectOutputStream {

        public AppendingObjectOutputStream(OutputStream out) throws IOException {
            super(out);
        }

        @Override
        protected void writeStreamHeader() throws IOException {
            // do not write a header, but reset:
            // this line added after another question
            // showed a problem with the original
            reset();
        }

    }

    private static <V> void save(Context context, String filename, ConcurrentHashMap<String, V> map, String key, V val) {
        try {

            map.put(key, val);
            ObjectOutputStream os = new AppendingObjectOutputStream(context.openFileOutput(filename, Context.MODE_APPEND));
            os.writeObject(key);
            os.writeObject(val);
            os.close();
        } catch (Exception e) {
            Log.e(TAG, "While saving results for " + key + " exception thrown: ", e);
        }
    }

    private void loadImageResults() {
        context.deleteFile("image_scenes");
        scenes = readObjectMap(context, IMAGE_SCENES_FILENAME);
        exifs = readObjectMap(context, IMAGE_EXIF_FILENAME);
        if (resetScenesModel) {
            scenes = new ConcurrentHashMap<>();
            context.deleteFile(IMAGE_SCENES_FILENAME);
        }
        try {
            if (scenes.isEmpty()) {
                ObjectOutputStream os = new ObjectOutputStream(context.openFileOutput(IMAGE_SCENES_FILENAME, Context.MODE_PRIVATE));
                os.writeObject("topCategories");
                os.close();
            }
            //exifs.clear();
            if (exifs.isEmpty()) {
                ObjectOutputStream os = new ObjectOutputStream(context.openFileOutput(IMAGE_EXIF_FILENAME, Context.MODE_PRIVATE));
                os.writeObject("exifs");
                os.close();
            }
        } catch (Exception e) {
            Log.e(TAG, "While creating empty files exception thrown: ", e);
        }
    }

    private synchronized SceneData classifyScenes(Bitmap bmp, StringBuilder text) {
        long startTime = SystemClock.uptimeMillis();
        Vector<float[]> embeds = pipe.pipe(bmp);
        Vector<Integer> ids = new Vector<>();
        for(float[] embed: embeds){
            int id = cluster.add(embed);
            ids.add(id);
        }
        long sceneTimeCost = SystemClock.uptimeMillis() - startTime;
        Log.i(TAG, "Timecost to run scene model inference: " + Long.toString(sceneTimeCost));
        text.append("Face and cluster:").append(sceneTimeCost).append(" ms\n");
        return new SceneData(ids);
    }


    public ImageAnalysisResults getImageAnalysisResults(String filename, Bitmap bmp, StringBuilder text,boolean needScene)
    {
        String key = getKey(filename);

        SceneData scene=null;
        if (!scenes.containsKey(key)) {
            if(needScene) {
                if (bmp == null)
                    bmp = loadBitmap(filename);
                scene = classifyScenes(bmp, text);
                for(Integer cluster: scene.scenes.categories){
                    file2cluster.put(filename, String.valueOf(cluster));
                    if(!cluster2files.containsKey(String.valueOf(cluster))){
                        cluster2files.put(String.valueOf(cluster), new HashSet<>());
                    }
                    cluster2files.get(String.valueOf(cluster)).add(filename);

                }
                save(context, IMAGE_SCENES_FILENAME, scenes, key, scene);
            }
        }
        else
            scene=scenes.get(key);

        EXIFData exifData=getEXIFData(filename);
        ImageAnalysisResults res = new ImageAnalysisResults(filename, scene,exifData);
        return res;
    }


    public ImageAnalysisResults getImageAnalysisResults(String filename) {
        StringBuilder text = new StringBuilder();
        return getImageAnalysisResults(filename,null,text,true);
    }

    private String getLocationDescription(double latitude, double longitude){
        String description=null;
        try {
            List<Address> addresses = geocoder.getFromLocation(latitude, longitude,1);
            StringBuilder text=new StringBuilder();
            if (addresses != null) {
                Address returnedAddress = addresses.get(0);
                String city=returnedAddress.getLocality();
                if (city!=null) {
                    text.append(city);
                }
                String countryName=returnedAddress.getCountryName();
                if(countryName!=null)
                    text.append(", ").append(countryName);
                /*
                for (int i = 0; i <= returnedAddress.getMaxAddressLineIndex(); i++) {
                    text.append(returnedAddress.getAddressLine(i));
                    if (i < returnedAddress.getMaxAddressLineIndex())
                        text.append(",");
                }*/
                description=text.toString();
                if(description.equals(""))
                    description=null;
            }
        }catch(Exception ex){
            //ignore
        }
        return description;
    }
    private String getKey(String filename){
        long dateModified = new File(filename).lastModified();
        String key = filename+"_"+dateModified;
        return key;
    }
    public EXIFData getEXIFData(String filename){
        String key =getKey(filename);
        EXIFData exifData;
        if(!exifs.containsKey(key)){
            exifData=new EXIFData(filename);
            if (exifData.latitude!=0 && exifData.longitude!=0) {
                exifData.description=getLocationDescription(exifData.latitude, exifData.longitude);
            }
            save(context, IMAGE_EXIF_FILENAME, exifs, key, exifData);
        }
        else{
            exifData=exifs.get(key);
            if (exifData.description==null && exifData.latitude!=0 && exifData.longitude!=0) {
                exifData.description = getLocationDescription(exifData.latitude, exifData.longitude);
                if(exifData.description!=null)
                    save(context, IMAGE_EXIF_FILENAME, exifs, key,exifData);
            }
        }
        return exifData;
    }
    public Bitmap loadBitmap(String fname) {
        Bitmap bmp = null;
        try {
            bmp = BitmapFactory.decodeFile(fname);
            EXIFData exifData=getEXIFData(fname);
            Matrix mat = new Matrix();
            switch (exifData.orientation) {
                case 6:
                    mat.postRotate(90);
                    break;
                case 3:
                    mat.postRotate(180);
                    break;
                case 8:
                    mat.postRotate(270);
                    break;
            }
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), mat, true);
        } catch (Exception e) {
            Log.e(TAG, "While loading image" + fname + " exception thrown: ", e);
        }
        return bmp;
    }

    private static DateFormat df=java.text.DateFormat.getDateInstance(DateFormat.MEDIUM,Locale.US);
    private String getDateFromTimeInMillis(long timeInMillis){
        Calendar calendar = Calendar.getInstance();
        calendar.setTimeInMillis(timeInMillis);
        return " "+df.format(calendar.getTime());
    }
    private void initPhotosTaken() {
        final String[] projection = {MediaStore.Images.Media.DATA, MediaStore.Images.Media.DATE_TAKEN};
        //String path= Environment.getExternalStorageDirectory().toString();//+"/DCIM/Camera";
        final String selection = null;//MediaStore.Images.Media.BUCKET_ID +" = ?";
        final String[] selectionArgs = null;//{String.valueOf(path.toLowerCase().hashCode())};
        photosTaken.clear();
        try {
            final Cursor cursor = context.getContentResolver().query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, projection, selection, selectionArgs, MediaStore.Images.ImageColumns.DATE_TAKEN + " DESC");
            if (cursor.moveToFirst()) {
                int dataColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
                int dateColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_TAKEN);
                do {
                    String data = cursor.getString(dataColumn);
                    String dateStr=cursor.getString(dateColumn);
                    if(dateStr==null)
                        dateStr="0";
                    Long dateCreated = Long.parseLong(dateStr);
                    photosTaken.put(data, dateCreated);

                    String strDate = getDateFromTimeInMillis(dateCreated);
                    if (!date2files.containsKey(strDate))
                        date2files.put(strDate, new HashSet<>());
                    date2files.get(strDate).add(data);
                    //Log.i(TAG, "load image: "+data);
                }
                while (cursor.moveToNext());
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "Exception thrown: " + e);
        }

        avgNumPhotosPerDay=0;
        for(Set<String> files : date2files.values())
            avgNumPhotosPerDay+=files.size();

        if(!date2files.isEmpty())
            avgNumPhotosPerDay/=date2files.size();
        if (avgNumPhotosPerDay<MIN_PHOTOS_PER_DAY)
            avgNumPhotosPerDay=MIN_PHOTOS_PER_DAY;
    }

    public Map<String,Long> getCameraImages() {
        return photosTaken;
    }

    private void addDayEvent(List<Map<String, Map<String, Set<String>>>> eventTimePeriod2Files, String category, String timePeriod, Set<String> filenames){
        int highLevelCategory=Integer.valueOf(category);
        if(highLevelCategory>=0) {
            Map<String,Map<String,Set<String>>> histo=eventTimePeriod2Files.get(highLevelCategory);
            if(!histo.containsKey(category))
                histo.put(category,new TreeMap<>(Collections.reverseOrder()));
            histo.get(category).put(timePeriod,filenames);

            //Log.d(TAG,"EVENTS!!! "+timePeriod+":"+category+" ("+highLevelCategory+"), "+filenames.size());
        }
    }
    public void updateSceneInEvents(List<Map<String, Map<String, Set<String>>>> eventTimePeriod2Files, String filename) {
        if (photosTaken.containsKey(filename)) {
            String strDate = getDateFromTimeInMillis(photosTaken.get(filename));
            if (date2files.containsKey(strDate)) {
                Set<String> files = date2files.get(strDate);

                if (files.size() < avgNumPhotosPerDay)
                    return;
                boolean allResultsAvailable = true;
                for (String f : files)
                    if (!file2Scene.containsKey(f)) {
                        allResultsAvailable = false;
                        break;
                    }
                if (!allResultsAvailable)
                    return;

                for (String cluster : cluster2files.keySet()) {
                    Set<String> cluster_files = cluster2files.get(cluster);
                    addDayEvent(eventTimePeriod2Files, cluster.toString(), strDate, cluster_files);
                }
            }
        }
    }
}
