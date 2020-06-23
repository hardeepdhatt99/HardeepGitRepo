package ca.utoronto.utm.paint;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.util.ArrayList;

public class Triangle extends shapes implements Command{
	private int size = 0;
	private ArrayList<Point> Points = new ArrayList<Point>();

	public Triangle() {
		
	}
	public Triangle(Point a, Point b, Point c, boolean fill) {
		this.Points.add(a);
		this.Points.add(b);
		this.Points.add(c);
		this.setfill(fill);
	}

	public Point getCornerOne() {
		return this.Points.get(0);
	}

	public void setCornerOne(Point cornerOne) {
		this.Points.set(0, cornerOne);
	}
	
	public Point getCornerTwo() {
		return this.Points.get(1);
	}

	public void setCornerTwo(Point cornerTwo) {
		this.Points.set(1, cornerTwo);
	}
	public Point getCornerThree() {
		return this.Points.get(2);
	}

	public void setCornerThree(Point cornerThree) {
		this.Points.set(2, cornerThree);
	}

	public void addPoint(Point point) {
		if (this.size<3) {
			this.Points.add(point);
			this.size++;
		}
	}
	public int getSize() {
		return this.size;
	}
	
	@Override
	public void execute(Graphics2D g2d) {
		int[] x = new int[3];
		int[] y = new int[3];
		for(int i = 0; i < Points.size(); i++) {
			x[i]=(Points.get(i).getX());
			y[i]=(Points.get(i).getY());
		}
		g2d.setColor(this.getColor());
		g2d.setStroke(new BasicStroke(this.getThickness()));
		g2d.setColor(this.getColor());
		g2d.setStroke(new BasicStroke(this.getThickness()));
		if (this.getfilled())g2d.fillPolygon(x, y, Points.size());
		else g2d.drawPolygon(x, y, 3);			
		}

}

