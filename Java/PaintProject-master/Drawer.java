package ca.utoronto.utm.paint;

import java.awt.Color;
import java.awt.event.MouseEvent;

public class Drawer {
	private ShapeManipulatorStrategy Strat;
	private shapes Shape;
	private PaintModel model;
	private boolean filled;
	private int thickness;
	private Color color;
	private Color background;
	private String Text;
	
	Drawer(ShapeManipulatorStrategy strat){
		this.setStrat(strat);
		filled = false;
		this.thickness = 1;
		this.color = Color.BLACK;
	}

	public void move(MouseEvent e) {
		this.Strat.move(e);
	}
	
	public void drag(MouseEvent e) {
		this.Strat.drag(e);
	}
	
	public void click(MouseEvent e) {
		this.Strat.click(e);
	}
	
	public void press(MouseEvent e) {
		this.Strat.press(e);
	}
	
	public void release(MouseEvent e) {
		this.Strat.release(e);
	}
	

	public ShapeManipulatorStrategy getStrat() {
		return Strat;
	}

	public void setStrat(ShapeManipulatorStrategy strat) {
		this.Strat = strat;
	}

	public shapes getShape() {
		return Shape;
	}

	public void setShape(shapes shape) {
		this.Shape = shape;
	}

	public PaintModel getModel() {
		return model;
	}

	public void setModel(PaintModel model) {
		this.model = model;
	}

	public boolean getFilled() {
		return this.filled;
	}
	
	public void setFilled(boolean b) {
		this.filled = b;
	}

	public int getThickness() {
		return this.thickness;
	}
	
	public void setThickness(int t) {
		this.thickness = t;
	}

	public Color getColor() {
		return this.color;
	}
	public void setColor(Color c) {
		this.color = c;
	}

	public void setBackground(Color c) {
		this.background = c;
		
	}
	public Color getBackground() {
		return this.background;
		
	}

	public void setText(String t) {
		this.Text = t;
	}
	
	public String getText() {
		return this.Text;
	}
	
}
