
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

public class Spider {

	public static void getRes(String URL) throws Exception {
		Document doc = null;
		Elements elems = null;
		String targetFile = "D:\\3D_list.txt";

		FileWriter writer = new FileWriter(targetFile, true);
		
		doc = Jsoup.connect(URL).timeout(5000).get();
		elems = doc.getElementsByClass("wqhgt");
		String launchDate = null;
		String seqNum = null;
		String result = null;
		String combinedInfo = null;
		for (int index = 2; index <= 21; index++) {
			Element ele = elems.select("tr").get(index);
			Elements inner_eles = ele.getElementsByTag("td");
			launchDate = inner_eles.get(0).text().toString();
			seqNum = inner_eles.get(1).text().toString();
			result = inner_eles.get(2).text().toString().replaceAll(" ", "");
			combinedInfo = launchDate + " " + seqNum + " " + result;
			writer.write(combinedInfo);
			writer.write("\n");
		}
		writer.close();

	}

	public static void main(String[] args) throws InterruptedException {
		// TODO Auto-generated method stub
		
		String url_head = "http://kaijiang.zhcw.com/zhcw/html/3d/list_";
		String url = null;
		String seq = null;
		try {
			for (int index = 1; index <= 254; index++) {
				seq = String.valueOf(index);
				url = url_head + seq + ".html";
				System.out.println(url);
				getRes(url);
				if (index % 8 == 0) {
					Thread.sleep(2000);
				}

			} // end loop

		} catch (Exception e) {
			System.err.println("go !: ");
			e.printStackTrace();
			System.exit(0);
		}

	}

}
