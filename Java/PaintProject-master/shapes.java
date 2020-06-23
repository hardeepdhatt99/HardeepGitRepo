package ca.utoronto.utm.paint;
//CuiSHiZhen
import java.awt.Color;
import java.awt.Graphics2D;

public class shapes implements Command{
private int thickness;
private Color color;
private boolean filled;

	public shapes(){
		filled = false;
}
	public void setfill(boolean filled){
		this.filled= filled;
	}
	public boolean getfilled(){
		return this.filled;
	}
	public void setColor(Color c){
		this.color=c;
	}
	public void setThickness(int t){
		this.thickness=t;
	}
	public int getThickness(){
		return this.thickness;
	}
	public Color getColor(){
		return this.color;
	}
	public String toText(){
		return "";
	}
	public String repeatStr(String str,int n){
		String s = "";
		for (int i=0;i<n;i++){
			s=s+str;
		}
		return s;
	}
	public void execute(Graphics2D g2d) {
		// Execute method implemented in each respective shape		
	}
}
