package ca.utoronto.utm.paint;

import java.awt.BasicStroke;
import java.awt.Graphics2D;
import java.util.ArrayList;

public class PolyLine extends shapes implements Command{
	private ArrayList<Point> points;
	private boolean skip = false;

	public PolyLine() {
		points = new ArrayList<Point>();
	}

	public ArrayList<Point> points() {
		return points;
	}

	public void AddPoint(Point point) {
		this.points.add(point);
		if (points.size()>1) {
			Point f = points.get(points.size()-1);
			Point s = points.get(points.size()-2);
			this.skip = (f.getX() == s.getX() && f.getY() == s.getY());
		}
	}
	
	public void removeLast() {
		this.points.remove(this.points.size()-1);
	}
	
	public void setFirstPoint(Point p) {
		if(this.size() == 0) this.points.add(p);
		
		else if(this.size() >= 1) this.points.set(0, p);
		}
	
	public Point getFirstPoint() {
		return this.points.get(0);
	}

	public void setSecondPoint(Point p) {
		if(this.size() == 1)this.points.add(p);
		else if(this.size() == 2) this.points.set(1, p);
	}
	
	public Point getSecondPoint() {
		return this.points.get(1);
	}
	
	public int size(){
		return this.points.size();
	}
	
	@Override
	public void execute(Graphics2D g2d) {
		for(int i=0;i<points.size()-1; i++){
			Point p1=points.get(i);
			Point p2=points.get(i+1);
			Point p3 = null;
			if(i<points.size()-2) {p3=points.get(i+2);}
			if((p1.getX() == -1 && p1.getY() == -1)||(p2.getX() == -1 && p2.getY() == -1)) i++;
			else if(p3 == null ||p2.getX() != p3.getX() || p2.getY() != p3.getY()) {
				g2d.drawLine(p1.getX(), p1.getY(), p2.getX(), p2.getY());
			}
			else {
				i++;
				i++;
			}
		}
		
	}

	public boolean getSkip() {
		return this.skip ;
	}
	
}