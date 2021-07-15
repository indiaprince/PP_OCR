using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using OpenCvSharp;
using System.IO;
using System.Reflection;
using Microsoft.Win32;

namespace OcrTest2
{
    public partial class MainWindow : System.Windows.Window
    {
        public static string assetsRelativePath;        // <- ocr directory
        public static string modelFilePath_det;
        public static string modelFilePath_rec;
        public static string imagesFolder_out;
        public static string dictFolder;
        public static string imagesFolder;
        public static string _predictSingleImage_re;                    // < - input image directory 

        double x, y; // Mouse Position 
        private bool isDrawing;
        private double start_x = 0;     //Crop Start Position X
        private double start_y = 0;     //Crop Start Position Y
        double cropped_width=0;
        double cropped_height= 0;
        int resize_h;
        int resize_w;
        public MainWindow()
        {
            InitializeComponent();

            assetsRelativePath = System.IO.Path.GetDirectoryName(Assembly.GetEntryAssembly()?.Location);
            modelFilePath_det = System.IO.Path.Combine(assetsRelativePath, "Models", "det11_ko.onnx");
            modelFilePath_rec = System.IO.Path.Combine(assetsRelativePath, "Models", "rec11_ko_v2.onnx");
            imagesFolder_out = System.IO.Path.Combine(assetsRelativePath, "Crop");
            dictFolder = System.IO.Path.Combine(assetsRelativePath, "Dict");
            imagesFolder = System.IO.Path.Combine(assetsRelativePath, "Input");
            _predictSingleImage_re = System.IO.Path.Combine(imagesFolder, "101.jpg");
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            /*
            OpenFileDialog a = new OpenFileDialog();
            if (a.ShowDialog() == true)
            {
                string fullpath = a.FileName;
                string filename = a.SafeFileName;
                string path = fullpath.Replace(filename, "");
                _predictSingleImage_re = path;
                string[] files = Directory.GetFiles(path);

            }
            */
            Img.Source = new BitmapImage(new Uri(_predictSingleImage_re));
            Image<Rgb24> image = SixLabors.ImageSharp.Image.Load<Rgb24>(_predictSingleImage_re, out IImageFormat format);

            int limit_side_len = 960;
            int h = image.Height;
            int w = image.Width;
            float ratio = 0;
            if (Math.Max(h, w) > limit_side_len)
            {
                if (h > w)
                    ratio = (float)limit_side_len / h;
                else
                    ratio = (float)limit_side_len / w;
            }
            else
                ratio = 1;

            resize_h = (int)(h * ratio);
            resize_w = (int)(w * ratio);

            resize_h = Math.Max((int)(Math.Round((double)resize_h / 32) * 32), 32);
            resize_w = Math.Max((int)(Math.Round((double)resize_w / 32) * 32), 32);
            Img.Width = resize_w;
            Img.Height = resize_h;

        }
        private void canvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            System.Windows.Point StartMousePoint = e.GetPosition(this.canvas);
            start_x = StartMousePoint.X;
            start_y = StartMousePoint.Y;
            isDrawing = true;
        }

        private void canvas_MouseMove(object sender, MouseEventArgs e)
        {
            System.Windows.Point currentMousePoint = e.GetPosition(this.canvas);
            if (isDrawing) 
                set_rect_bound(start_x, start_y, currentMousePoint.X, currentMousePoint.Y);
            canvas.Children.Remove(Rect);
        }

        private void canvas_MouseUp(object sender, MouseButtonEventArgs e)
        {
            System.Windows.Point EndMousePoint = e.GetPosition(this.canvas);
            isDrawing = false;
        }
        void set_rect_bound(double start_x,double start_y,double cur_x,double cur_y)            //Setting Cropped Area
        {
            x = Math.Min(start_x, cur_x);
            y = Math.Min(start_y, cur_y);
            cropped_width = Math.Abs(cur_x - start_x);
            cropped_height = Math.Abs(cur_y - start_y);
            Rect.Margin = new Thickness(x,y,0,0);
            Rect.Width = cropped_width;
            Rect.Height = cropped_height;     
            
        }

        private void BtnCrop_Click(object sender, RoutedEventArgs e)            // When Crop Button Clicked, Save Cropped Area into Local
        {
            CropRstCanvas.Visibility = Visibility.Visible;
            DetectRstDock.Visibility = Visibility.Collapsed;
            BitmapImage fullBitmap = new BitmapImage();
            fullBitmap.BeginInit();
            //fullBitmap = new BitmapImage(new Uri(_predictSingleImage_re));
            fullBitmap.UriSource = new Uri(_predictSingleImage_re);
            fullBitmap.DecodePixelWidth = resize_w;
            fullBitmap.DecodePixelHeight = resize_h;
            fullBitmap.EndInit();
            Src.Source = new CroppedBitmap(fullBitmap, new Int32Rect((int)x,(int)y,(int)cropped_width, (int)cropped_height));
            Save_Img(Src);
        }
        private void BtnDet_Click(object sender, RoutedEventArgs e)
        {
            CropRstCanvas.Visibility = Visibility.Collapsed;
            DetectRstDock.Visibility = Visibility.Visible;
            // image resize
            
            Image<Rgb24> image = SixLabors.ImageSharp.Image.Load<Rgb24>(_predictSingleImage_re, out IImageFormat format);
            int h = image.Height;
            int w = image.Width;
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new SixLabors.ImageSharp.Size((int)resize_w, (int)resize_h),
                    Mode = SixLabors.ImageSharp.Processing.ResizeMode.Stretch
                });
            });
            float ratio_h = resize_h / (float)h;
            float ratio_w = resize_w / (float)w;
            //image resize


            // normalizing
            int batch = 1;
            int channel = 3;
            int height = resize_h;
            int width = resize_w;
            Tensor<float> input = new DenseTensor<float>(new[] { batch, channel, height, width });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            for (int k = 0; k < batch; k++)
            {
                for (int y = 0; y < height; y++)
                {
                    Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                    for (int x = 0; x < width; x++)
                    {
                        input[k, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                        input[k, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                        input[k, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                    }
                }
            }
            // normalizing


            //inferences
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("x", input) };
            var inference = new InferenceSession(modelFilePath_det);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = inference.Run(inputs);
            var resultsArray = results.ToArray();
            float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
            //inference


            //threshold
            float[,] box2 = new float[resize_h, resize_w];
            
            for (int i = 0; i < resize_h; i++)
            {
                for (int j = 0; j < resize_w; j++)
                {
                    //box2[i, j] = boxes[resize_w * i + j];
                    if (boxes[resize_w * i + j] > 0.3f) box2[i, j] = 0.999f;
                    else box2[i, j] = 0.0f;
                }
            }
            //threshold
            Console.WriteLine("===========================");


            
            Mat img = Cv2.ImRead(_predictSingleImage_re);
            Cv2.Resize(img, img, new OpenCvSharp.Size(resize_w, resize_h));
            OpenCvSharp.Point[][] contours;
            HierarchyIndex[] hierarchy;
            Mat src2 = new Mat(resize_h, resize_w, MatType.CV_8UC1);
            Mat tmp = new Mat();
            for (int i = 0; i < resize_h; i++)
            {
                for (int j = 0; j < resize_w; j++)
                {
                    src2.At<float>(i, j) = box2[i, j];
                }
            }
            src2.CopyTo(tmp);

            // First contour
            Cv2.FindContours(src2, out contours, out hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
            Point2f[] points;
            Polygon poly = null;
            List<Polygon> bx = new List<Polygon>();
            List<List<OpenCvSharp.Point>> pts = new List<List<OpenCvSharp.Point>>();
            List<OpenCvSharp.Point> pt = new List<OpenCvSharp.Point>();
            for (int i = 0; i < contours.Length; i++)
            {
                pt.Clear();
                Cv2.DrawContours(src2, contours, i,2);
                RotatedRect rect = Cv2.MinAreaRect(contours[i]);
                if (rect.Size.Width < 3) continue;
                poly = null;
                points = Cv2.BoxPoints(rect);
                
                float dist = ((float)Cv2.ContourArea(contours[i]) * 1.6f) / (float)Cv2.ArcLength(contours[i],true);

                Console.Write("dist = ");
                Console.WriteLine(dist);
                Console.Write("area = ");
                Console.WriteLine((float)Cv2.ContourArea(contours[i]));
                Console.Write("perimeter = ");
                Console.WriteLine((float)Cv2.ArcLength(contours[i], true));

                PointCollection myPointCollection = new PointCollection();

                myPointCollection.Add(new System.Windows.Point(points[0].X- dist, points[0].Y- dist));
                myPointCollection.Add(new System.Windows.Point(points[1].X+ dist, points[1].Y- dist));
                myPointCollection.Add(new System.Windows.Point(points[2].X+ dist, points[2].Y+ dist));
                myPointCollection.Add(new System.Windows.Point(points[3].X- dist, points[3].Y+ dist));
                pt.Add(new OpenCvSharp.Point(points[0].X - dist, points[0].Y - dist));
                pt.Add(new OpenCvSharp.Point(points[1].X + dist, points[1].Y - dist));
                pt.Add(new OpenCvSharp.Point(points[2].X + dist, points[2].Y + dist));
                pt.Add(new OpenCvSharp.Point(points[3].X - dist, points[3].Y + dist));
                pts.Add(pt);
                Cv2.Polylines(tmp, pts, true, Scalar.White, 2);
                poly = CreatePoly(myPointCollection, Brushes.Red, 2, null);
                //canvas.Children.Add(poly);
            }
            // First contour
            // Second contour
            Polygon poly2;
            OpenCvSharp.Point[][] contours2;
            HierarchyIndex[] hierarchy2;
            Polygon rect_final; 

            Cv2.FindContours(tmp, out contours2, out hierarchy2, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            Point2f[] points2;
            for (int i = 0; i < contours2.Length; i++)
            {
                Cv2.DrawContours(tmp, contours2, i, Scalar.White , -1);
                RotatedRect rect2 = Cv2.MinAreaRect(contours2[i]);
                if (rect2.Size.Width < 5) continue;
                points2 = Cv2.BoxPoints(rect2);

                poly2 = null;
                rect_final = null;
                PointCollection final_box = new PointCollection();
                PointCollection final_rect_pt = new PointCollection();

                final_box.Add(new System.Windows.Point(points2[0].X, points2[0].Y ));
                final_box.Add(new System.Windows.Point(points2[1].X, points2[1].Y ));
                final_box.Add(new System.Windows.Point(points2[2].X, points2[2].Y ));
                final_box.Add(new System.Windows.Point(points2[3].X, points2[3].Y ));

                // rather organizing each points, just tried to find LT point and RB point
                float minx = Math.Min(Math.Min(points2[0].X, points2[1].X), Math.Min(points2[2].X, points2[3].X));      
                float miny = Math.Min(Math.Min(points2[0].Y, points2[1].Y), Math.Min(points2[2].Y, points2[3].Y));
                float maxx = Math.Max(Math.Max(points2[0].X, points2[1].X), Math.Max(points2[2].X, points2[3].X));
                float maxy = Math.Max(Math.Max(points2[0].Y, points2[1].Y), Math.Max(points2[2].Y, points2[3].Y));

                final_rect_pt.Add(new System.Windows.Point(minx, miny));
                final_rect_pt.Add(new System.Windows.Point(maxx, miny));
                final_rect_pt.Add(new System.Windows.Point(maxx, maxy));
                final_rect_pt.Add(new System.Windows.Point(minx, maxy));


                poly2 = CreatePoly(final_box, Brushes.Blue, 2, null);
                rect_final = CreatePoly(final_rect_pt, Brushes.Black, 2, null);

                //canvas.Children.Add(poly2);
                canvas.Children.Add(rect_final);
                bx.Add(rect_final);

                Cv2.Rectangle(img,new OpenCvSharp.Point(points2[1].X, points2[1].Y), new OpenCvSharp.Point(points2[3].X, points2[3].Y) , Scalar.Black,2);// 사각형으로 그리기, 사각형 하면 될듯.
            }
            //Cv2.ImShow("2", img);
            DetectRstDock.Children.Clear();

            Console.WriteLine("===========================");
            //Recognition
            double max_wh_ratio = 0;

            for (int i = bx.Count-1; i >=0 ; i--)
            {
                string result = "";
                var rstImg = new System.Windows.Controls.Image
                {
                    HorizontalAlignment = HorizontalAlignment.Left,
                    Height=25,
                    VerticalAlignment = VerticalAlignment.Center
                };
                var rstTxt = new TextBlock();
                var subPanel = new StackPanel 
                {
                    Orientation = Orientation.Vertical
                };
                subPanel.SetValue(DockPanel.DockProperty, Dock.Top);
                subPanel.Children.Add(rstImg);
                subPanel.Children.Add(rstTxt);

                DetectRstDock.Children.Add(subPanel);
                BitmapImage fullBitmap= new BitmapImage();
                fullBitmap.BeginInit();
                //fullBitmap = new BitmapImage(new Uri(_predictSingleImage_re));
                fullBitmap.UriSource = new Uri(_predictSingleImage_re);
                fullBitmap.DecodePixelWidth = resize_w; 
                fullBitmap.DecodePixelHeight = resize_h;
                fullBitmap.EndInit();

                rstImg.Source = new CroppedBitmap(fullBitmap, new Int32Rect((int)bx[i].Points[0].X, (int)bx[i].Points[0].Y, Math.Abs((int)bx[i].Points[1].X - (int)bx[i].Points[0].X), Math.Abs((int)bx[i].Points[2].Y - (int)bx[i].Points[1].Y)));
                Save_Img(rstImg);
                Image<Rgb24> image_rec = SixLabors.ImageSharp.Image.Load<Rgb24>(imagesFolder_out + "\\cropped.jpg", out IImageFormat format1);
                double hg = image_rec.Height;
                double wd = image_rec.Width;
                Console.WriteLine(image_rec.Size());
                double wh_ratio = wd * 1.0 / hg;
                int imgH = 32;
                int imgW = 320;
                max_wh_ratio = Math.Max(max_wh_ratio, wh_ratio);

                imgW = (int)(32 * max_wh_ratio);
                double ratio = wd / (double)hg;
                int resized_rec_w = 0;
                if (Math.Ceiling(imgH * ratio) > imgW) resized_rec_w = imgW;
                else resized_rec_w = (int)Math.Ceiling(imgH * ratio);

                image_rec.Mutate(x => x.Resize(resized_rec_w, 32));
                Console.WriteLine(image_rec.Size());
                int batch_r = 1;
                int channel_r = 3;
                int height_r = imgH;
                int width_r = imgW;
                Tensor<float> input_r = new DenseTensor<float>(new[] { batch_r, channel_r, height_r, width_r });

                for (int y = 0; y < height_r; y++)
                {
                    Span<Rgb24> pixelSpan = image_rec.GetPixelRowSpan(y);
                    for (int x = 0; x < width_r; x++)
                    {
                        
                        if (x < resized_rec_w)
                        {
                            input_r[0, 0, y, x] = ((pixelSpan[x].R / 255f) - (float)0.5) / (float)0.5;
                            input_r[0, 1, y, x] = ((pixelSpan[x].G / 255f) - (float)0.5) / (float)0.5;
                            input_r[0, 2, y, x] = ((pixelSpan[x].B / 255f) - (float)0.5) / (float)0.5;
                            continue;
                        }
                        input_r[0, 0, y, x] = 0;
                        input_r[0, 1, y, x] = 0;
                        input_r[0, 2, y, x] = 0;
                    }
                }
                var inputs_r = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("x", input_r) };
                var inference_r = new InferenceSession(modelFilePath_rec);
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results_r = inference_r.Run(inputs_r);
                var resultsArray_r = results_r.ToArray();
                Console.WriteLine(resultsArray_r.Length);
                float[] d = resultsArray_r[0].AsEnumerable<float>().ToArray();
                Console.WriteLine(d.Length);
                int len = d.Length / 3689;
                double[] argmax = new double[len];
                int[] index = new int[len];
                double[,] res = new double[len, 3689];

                for (int k = 0; k < len; k++)
                {
                    int idx = 0;
                    double max = -10.0;
                    for (int j = 0; j < 3689; j++)
                    {  
                        res[k, j] = (double)d[3689 * k + j];
                        if (max < res[k, j])
                        {
                            idx = j;
                            max = res[k, j];
                        }
                    }
                    index[k] = idx;
                    argmax[k] = max;
                    Console.Write(index[k]);
                    Console.Write("     ");

                    Console.WriteLine(argmax[k]);
                }
                string line;
                for (int k = 0; k < len; k++)
                {
                    if (index[k] == 0)
                        continue;
                   
                    if (k > 0 && index[k] == index[k - 1])
                        continue;
                
                    int count = 0;
                    StreamReader file = new StreamReader(dictFolder + "\\korean_dict.txt");
                    while ((line = file.ReadLine()) != null)
                    {
                        count++;

                        if (count == index[k])
                        {
                            result = result + line;
                            Console.WriteLine(line);
                            Console.WriteLine(count);
                        }
                    }
                }
                rstTxt.FontSize = 30;
                rstTxt.Text = result;
            }
            //Recognition
        }
        private void BtnRec_Click(object sender, RoutedEventArgs e)
        {
            CropRstCanvas.Visibility = Visibility.Visible;
            DetectRstDock.Visibility = Visibility.Collapsed;

            Image<Rgb24> image_rec = SixLabors.ImageSharp.Image.Load<Rgb24>(imagesFolder_out + "\\cropped.jpg", out IImageFormat format1);
            double h = image_rec.Height;
            double w = image_rec.Width;
            Console.WriteLine(image_rec.Size());

            int imgH = 32;
            int imgW = 320;
            double ratio = w / (double)h;
            int resized_w = 0;
            if (Math.Ceiling(imgH * ratio) > imgW) resized_w = imgW;
            else resized_w = (int)Math.Ceiling(imgH * ratio);

            image_rec.Mutate(x => x.Resize(resized_w, 32));
            Console.WriteLine(image_rec.Size());
            int batch_r = 1;
            int channel_r = 3;
            int height_r = imgH;
            int width_r = imgW;
            Tensor<float> input_r = new DenseTensor<float>(new[] { batch_r, channel_r, height_r, width_r });

            for (int y = 0; y < height_r; y++)
            {
                Span<Rgb24> pixelSpan = image_rec.GetPixelRowSpan(y);
                for (int x = 0; x < width_r; x++)
                {
                    if (x >= resized_w)
                    {
                        input_r[0, 0, y, x] = 0;
                        input_r[0, 1, y, x] = 0;
                        input_r[0, 2, y, x] = 0;
                        continue;
                    }
                    input_r[0, 0, y, x] = ((pixelSpan[x].R / 255f) - (float)0.5) / (float)0.5;
                    input_r[0, 1, y, x] = ((pixelSpan[x].G / 255f) - (float)0.5) / (float)0.5;
                    input_r[0, 2, y, x] = ((pixelSpan[x].B / 255f) - (float)0.5) / (float)0.5;
                }
            }
            var inputs_r = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("x", input_r) };
            var inference_r = new InferenceSession(modelFilePath_rec);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results_r = inference_r.Run(inputs_r);
            var resultsArray_r = results_r.ToArray();
            float[] d = resultsArray_r[0].AsEnumerable<float>().ToArray();
            Console.Write(d.Length);
            double[] argmax = new double[80];
            int[] index = new int[80];
            double[,] res = new double[80, 3689];
            for (int i = 0; i < 80; i++)
            {
                int idx = 0;
                double max = -1;
                for (int j = 0; j < 3689; j++)
                {
                    res[i, j] = (double)d[3689 * i + j];
                    if (max < res[i, j])
                    {
                        idx = j;
                        max = res[i, j];
                    }
                }
                index[i] = idx;
                argmax[i] = max;
            }
            string line;
            string result = "";
            for (int i = 0; i < 80; i++)
            {
                if (index[i] == 0) continue;
                if (i>0 && index[i] == index[i - 1]) continue;
                int count = 0;
                System.IO.StreamReader file = new System.IO.StreamReader(dictFolder + "\\korean_dict.txt");
                while ((line = file.ReadLine()) != null)
                {
                    count++;
                    if (count == index[i])
                    {
                        result = result + line;
                    }
                }
            }
            Result.FontSize = 30;
            Result.Text = result;
        }

        public void Save_Img(System.Windows.Controls.Image image)
        {
            if (!Directory.Exists(imagesFolder_out))
                Directory.CreateDirectory(imagesFolder_out);

            var path = imagesFolder_out + "\\cropped.jpg";

            var encoder = new JpegBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create((BitmapSource)image.Source));
            using (FileStream stream = new FileStream(path, FileMode.Create))
                encoder.Save(stream);  
        }
        public Polygon CreatePoly(PointCollection pts, Brush brush, double thickness, DoubleCollection dashStyle){
            Polygon poly = new Polygon();
            poly.Points = pts;
            poly.Stroke = brush;
            poly.StrokeThickness = thickness;
            poly.StrokeDashArray = dashStyle;
            return poly;
        }

    }
}
