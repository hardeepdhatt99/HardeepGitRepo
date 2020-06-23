package ca.utoronto.utm.paint;

import java.awt.BasicStroke;
import java.awt.Graphics2D;
import java.util.ArrayList;

public class Eraser extends Squiggle implements Command{
	private ArrayList<Point> points;

	public Eraser() {
		points = new ArrayList<Point>();
	}

	public ArrayList<Point> points() {
		return points;
	}

	public void AddPoint(Point point) {
		this.points.add(point);
	}
	
	public Point getLast() {
		return this.points.get(this.points.size()-1);
	}
	
	@Override
	public void execute(Graphics2D g2d) {
		int i = 0;
		if (this.points().size()>2){
		while(i<this.points().size()-2){
			g2d.setColor(this.getColor());
			g2d.setStroke(new BasicStroke(this.getThickness()));
			g2d.drawLine(this.points().get(i).x, this.points().get(i).y, this.points().get(i+1).x, this.points().get(i+1).y);
			i++;
		}}
	}
}