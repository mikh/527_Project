const int SIZE = 28 * 28;
const int SAMPLES = 1000;

private void ReadData(int n) {
  byte[] bytes = new byte[SIZE * SAMPLES];
  FileStream fs = File.OpenRead("data"+n);
  try {
    fs.Read(bytes, 0, SIZE * SAMPLES);
    fs.Close();
  }
  catch (Exception ex) {
    MessageBox.Show(ex.ToString());
  }
  finally {
    fs.Close();
  }
  for (int i = 0; i < SAMPLES; i++) {
    try {
      Bitmap temp = new Bitmap(28, 28);
      for (int j = 0; j < 28; j++) {
        for (int k = 0; k < 28; k++) {
          byte yo = (byte)(0xff - bytes[i * SIZE + 28 * k + j]);
          temp.SetPixel(j, k, Color.FromArgb(yo, yo, yo));
        }
      }
      temp.Save(n+"\\data"+n+"_" + i + ".jpg", ImageFormat.Jpeg);
    }
    catch (Exception ex) {
      string msg = ex.ToString();
      MessageBox.Show(msg);
    }
    progressBar1.Value = i;
  }
}

private void button1_Click(object sender, EventArgs e) {
  if (Directory.Exists(textBox1.Text)) {
    Directory.Delete(textBox1.Text, true);
  }
  Directory.CreateDirectory(textBox1.Text);
  int n = System.Convert.ToInt32(textBox1.Text);
  ReadData(n);
  MessageBox.Show("Done");
  progressBar1.Value = 0;
}

private void Form1_Load(object sender, EventArgs e) {
  progressBar1.Maximum = SAMPLES;
}