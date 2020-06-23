package ca.utoronto.utm.paint;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;

public class Square extends shapes implements Command{
	private Point POne;
	private Point PTwo;
	private int height;
	
	public Square(Point POne, Point PTwo, int height){
		this.POne = POne;
		this.PTwo = PTwo;
		this.height = height;
	}

	public Point getPOne() {
		return POne;
	}

	public void setPOne(Point POne) {
		this.POne = POne;
	}

	public Point getPTwo() {
		return PTwo;
	}

	public void setPTwo(Point PTwo) {
		this.PTwo = PTwo;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
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
				this.POne.x+
				super.repeatStr("N", (5-(""+this.POne.x).length()))+
				this.POne.y+
				super.repeatStr("N", (5-(""+this.POne.y).length()))+
				this.PTwo.x+
				super.repeatStr("N", (5-(""+this.PTwo.x).length()))+
				this.PTwo.y+
				super.repeatStr("N", (5-(""+this.PTwo.y).length()))+
				this.height;
	    
	}
	@Override
	public void execute(Graphics2D g2d) {
		int x = 0;
		int y = 0;
		int width = this.getPTwo().getX()-this.getPOne().getX();
		int height = this.getPTwo().getY()-this.getPOne().getY();
		int w = Math.abs(width);
		int d = Math.abs(height);
		int measure = Math.max(w, d);
		if(height >= 0 && width >= 0) {//square down right from first click
			x = this.getPOne().getX();
			y = this.getPOne().getY();
		}
		else if(height >= 0 && width < 0){//square down left from first click
			x = this.getPOne().getX() - measure;
			y = this.getPOne().getY();
		}
		else if(height < 0 && width >= 0){//square up right from first click
			x = this.getPOne().getX();
			y = this.getPOne().getY() - measure;
		}
		else if(height < 0 && width < 0) {//square up left from first click
			x = this.getPOne().getX() - measure;
			y = this.getPOne().getY() - measure;
		}
		g2d.setColor(this.getColor());
		g2d.setStroke(new BasicStroke(this.getThickness()));
		if (this.getfilled()) g2d.fillRect(x, y, measure, measure);
		else g2d.drawRect(x, y, measure, measure);
	
	}
}
