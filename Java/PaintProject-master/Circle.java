package ca.utoronto.utm.paint;

import java.awt.BasicStroke;
import java.awt.Graphics2D;

public class Circle extends shapes implements Command{
	private Point centre;
	private Point topLeft = new Point(0,0);
	private int radius;
	
	public Circle(Point centre, int radius){
		this.centre = centre;
		this.radius = radius;
		setTopLeftIndirect(this.centre);
	}
	public Point getCentre() {
		return centre;
	}

	public void setCentre(Point centre) {
		this.centre = centre;
	}

	public Point getTopLeft() {
		return topLeft;
	}
	
	public void setTopLeftIndirect(Point centre) {
		this.topLeft.setX(this.centre.getX() - radius);
		this.topLeft.setY(this.centre.getY() - radius);
	}
	
	public void setTopLeftDirect(Point centre) {
		this.topLeft.setX(this.centre.getX()); 
		this.topLeft.setY(this.centre.getY()); 
	}
	public int getRadius() {
		return radius;
	}

	public void setRadius(int radius) {
		this.radius = radius;
		setTopLeftIndirect(this.centre);
	}

	public void setRadiusP(int x, int y) {
		int xValues = this.getCentre().getX() - x; 						//distance from x to x
		xValues = xValues * xValues;									//square the distance
		int yValues = (this.getCentre().getY() - y); 					//distance from y to y
		yValues = yValues * yValues;									//square the distance
		this.radius = (int) (Math.sqrt(xValues + yValues));				//sqrt the values 
		setTopLeftIndirect(this.centre);
	}
	public String toText(){
		return "C"
				+this.getThickness()
				+super.repeatStr("N", (5-(""+this.getThickness()).length()))
				+this.getColor().getRed()
				+super.repeatStr("N", (5-(""+this.getColor().getRed()).length()))
				+this.getColor().getGreen()
				+super.repeatStr("N", (5-(""+this.getColor().getGreen()).length()))
				+this.getColor().getBlue()
				+super.repeatStr("N", (5-(""+this.getColor().getBlue()).length()))
				+this.centre.x
				+super.repeatStr("N", (5-(""+this.centre.x).length()))
				+this.centre.y
				+super.repeatStr("N", (5-(""+this.centre.y).length()))
				+this.getRadius();
	}
	@Override
	public void execute(Graphics2D g2d) {
		int x = 0;
		int y = 0;
		x = this.getTopLeft().getX();
		y = this.getTopLeft().getY();
		int radius = this.getRadius();
		g2d.setColor(this.getColor());
		g2d.setStroke(new BasicStroke(this.getThickness()));
		if (this.getfilled())g2d.fillOval(x, y, 2*radius, 2*radius);
		else g2d.drawOval(x, y, 2*radius, 2*radius);
	}
}
