package ca.utoronto.utm.paint;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;

public class Rectangle extends shapes implements Command{
	private Point cornerOne;
	private Point cornerTwo;
	private int height;
	private int width;	
	
	public Rectangle(Point cornerOne, Point cornerTwo, int height, int width){
		this.cornerOne = cornerOne;
		this.cornerTwo = cornerTwo;
		this.height = height;
		this.width = width;
	}

	public Rectangle() {
		// TODO Auto-generated constructor stub
	}

	public Point getCornerOne() {
		return cornerOne;
	}

	public void setCornerOne(Point cornerOne) {
		this.cornerOne = cornerOne;
	}
	
	public Point getCornerTwo() {
		return cornerTwo;
	}

	public void setCornerTwo(Point cornerTwo) {
		this.cornerTwo = cornerTwo;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}
	
	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}
	public String toText(){
		return "S"+
				+this.getThickness()
				+super.repeatStr("N", (5-(""+this.getThickness()).length()))
				+this.getColor().getRed()
				+super.repeatStr("N", (5-(""+this.getColor().getRed()).length()))
				+this.getColor().getGreen()
				+super.repeatStr("N", (5-(""+this.getColor().getGreen()).length()))
				+this.getColor().getBlue()
				+super.repeatStr("N", (5-(""+this.getColor().getBlue()).length()))+
				this.cornerOne.x+
				super.repeatStr("N", (5-(""+this.cornerOne.x).length()))+
				this.cornerOne.y+
				super.repeatStr("N", (5-(""+this.cornerOne.y).length()))+
				this.cornerTwo.x+
				super.repeatStr("N", (5-(""+this.cornerTwo.x).length()))+
				this.cornerTwo.y+
				super.repeatStr("N", (5-(""+this.cornerTwo.y).length()))+
				this.height+
				super.repeatStr("N", (5-(""+this.height).length()))+
				this.width;
	    
	}
	@Override
	public void execute(Graphics2D g2d) {
		int x = 0;
		int y = 0;
		x = Math.min(this.getCornerOne().getX(), this.getCornerTwo().getX());
		y = Math.min(this.getCornerOne().getY(), this.getCornerTwo().getY());
		int height = Math.abs(this.getCornerOne().getX() - this.getCornerTwo().getX());
		int width = Math.abs(this.getCornerOne().getY() - this.getCornerTwo().getY());
		g2d.setColor(this.getColor());
		g2d.setStroke(new BasicStroke(this.getThickness()));
		if (this.getfilled())g2d.fillRect(x, y, height, width);
		else g2d.drawRect(x, y, height, width);
			
		}
}

