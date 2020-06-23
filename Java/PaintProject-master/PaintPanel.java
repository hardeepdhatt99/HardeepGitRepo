package ca.utoronto.utm.paint;

import javax.swing.*;  
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.ArrayList;
import java.util.Observable;
import java.util.Observer;

// https://docs.oracle.com/javase/8/docs/api/java/awt/Graphics2D.html
// https://docs.oracle.com/javase/tutorial/2d/

class PaintPanel extends JPanel implements Observer, MouseMotionListener, MouseListener  {
	private PaintModel model; // slight departure from MVC, because of the way painting works
	private View view; // So we can talk to our parent or other components of the view
	
	private Drawer drawer;
	private boolean getFilled;
	private Color color = Color.BLACK;
	private String t = "";
	private String mode; // modifies how we interpret input (could be better?)
	
	public Color getColor() {
		return this.color;
	}
	
	public PaintPanel(PaintModel model, View view){
		this.drawer = new Drawer(null);
		this.drawer.setStrat(new makeCircle(this.drawer));
		this.drawer.setModel(model);
		this.drawer.setBackground(Color.WHITE);
		this.getFilled = false;
		this.setBackground(Color.WHITE);
		this.setPreferredSize(new Dimension(300,300));
		this.addMouseListener(this);
		this.addMouseMotionListener(this);
		
		this.model = model;
		this.model.addObserver(this);
		
		this.view=view;
	}

	public boolean getFill() {
		return this.getFilled;
	}
	
	public void toggleFill() {
		this.getFilled = !this.getFilled;
		this.drawer.setFilled(this.getFilled);
	}

	/**
	 *  View aspect of this
	 */
	public void paintComponent(Graphics g) {
		// Use g to draw on the JPanel, lookup java.awt.Graphics in
		// the javadoc to see more of what this can do for you!!
		
        super.paintComponent(g); //paint background
        Graphics2D g2d = (Graphics2D) g; // lets use the advanced api
        //setBackground(Color.white);
		// Origin is at the top left of the window 50 over, 75 down
		g2d.setColor(Color.BLACK);
		
		for (Command c: this.model.getDrawingCommands()) {
			c.execute(g2d);			
		}
		
		for (Command c: this.model.getTempDrawing()) {
			c.execute(g2d);	
		}
		this.model.callHistory();
		
		g2d.dispose();
	}
	
	@Override
	public void update(Observable o, Object arg) {
		// Not exactly how MVC works, but similar.

		if (o.getClass()== ThicknessListenerAndModel.class){
			this.drawer.setThickness((int) arg);
		}
		if (o.getClass()== DataColorPanel.class){
			this.color = (Color)arg;
			this.drawer.setColor((Color)arg);
		}
		
		if (o.getClass()== TextBoxPanelTalksToPaintModel.class) {
			this.t = (String)arg;
			this.drawer.setText(t);
		}
		this.repaint(); // Schedule a call to paintComponent
	}
	
	/**
	 *  Controller aspect of this
	 */
	public void setMode(String mode){

		this.drawer.setColor(this.color);
		if (mode != "Polyline" && this.mode == "Polyline") this.model.resetTemp();
		if (mode != "Triangle" && this.mode == "Triangle") this.model.resetTemp();
		
		switch(mode) {
		case "Circle":
			this.drawer.setStrat(new makeCircle(this.drawer));
			break;
		case "Rectangle":
			this.drawer.setStrat(new makeRectangle(this.drawer));
			break;
		case "Square":
			this.drawer.setStrat(new makeSquare(this.drawer));
			break;
		case "Triangle":
			this.drawer.setStrat(new makeTriangle(this.drawer));
			break;
		case "Squiggle":
			this.drawer.setStrat(new makeSquiggle(this.drawer));
			break;
		case "Erase":
			this.drawer.setStrat(new makeEraser(this.drawer));
			break;
		case "Polyline":
			this.drawer.setStrat(new makePolyline(this.drawer));
			break;
		case "Text":
			this.drawer.setStrat(new makeText(this.drawer));
			break;
		default:
			System.out.println("default");
			break;
		}
		this.mode=mode;
		this.repaint();
	}
	
	public void clear() {
		this.model.emptyStacks();
		super.paintComponent(getGraphics());
	}
	
	// MouseMotionListener below
	@Override
	public void mouseMoved(MouseEvent e) {
		this.drawer.move(e);
	}
	@Override
	public void mouseDragged(MouseEvent e) {
		this.drawer.drag(e);
	}

	// MouseListener below
	@Override
	public void mouseClicked(MouseEvent e) {
		this.drawer.click(e);
	}
	
	@Override
	public void mousePressed(MouseEvent e) {
		this.drawer.press(e);
	}

	@Override
	public void mouseReleased(MouseEvent e) {
		this.drawer.release(e);
		this.model.resetTemp();
	}
 
	@Override
	public void mouseEntered(MouseEvent e) {
	}

	@Override
	public void mouseExited(MouseEvent e) {
	}
}
