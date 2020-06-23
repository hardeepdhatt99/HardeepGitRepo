package ca.utoronto.utm.paint;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;

public class Text extends shapes implements Command{
	private String text;
	private Point point;
	private int size = 20;
	private Color color;
	
	public Text(String text, Point point) {
		this.text = text;
		this.point = point;
	}
	
	public void setText(String s) {
		this.text = s;
	}
	
	public void setPoint(Point p) { 
		this.point = p;
	}
	
	public void setSize(int s) {
		this.size = s * 4;
	}
	
	public String getText() {
		return this.text;
	}
	 
	public Point getPoint() {
		return this.point;
	}
	
	public void setColor(Color c) {
		this.color = c;
	}
	
	@Override
	public void execute(Graphics2D g2d) {
		g2d.setFont(new Font("TimesRoman", Font.PLAIN, this.size)); 
		g2d.setColor(this.color);
		try {
			g2d.drawChars(this.text.toCharArray(), 0, this.text.length(), this.point.getX(), this.point.getY());
		} catch (NullPointerException e) {
			System.out.println("No text to add");
		}
	}
}
