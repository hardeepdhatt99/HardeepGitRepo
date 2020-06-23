package ca.utoronto.utm.paint;

import java.awt.Color;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.Observable;
import java.util.Stack;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileSystemView;

public class PaintModel extends Observable {
	private ArrayList<Point> points=new ArrayList<Point>();
	private ArrayList<Point> lines=new ArrayList<Point>();
	private ArrayList<Point> tLines=new ArrayList<Point>();
	private shapes lastChanged;
	private Stack<shapes> DrawingCommands = new Stack<shapes>();
	private Stack<shapes> tempDrawing = new Stack<shapes>();
	private Stack<shapes> UndoCommands = new Stack<shapes>();
	private Stack<String> Order = new Stack<String>();
	
	public void addPoint(Point p){
		this.points.add(p);
		this.setChanged();
		this.notifyObservers();
	}
	
	public ArrayList<Point> getPoints(){
		return points;
	}
	
	public void addLinePoint(Point p){
		this.lines.add(p);
		this.setChanged();
		this.notifyObservers();
	}
	
	public ArrayList<Point> getLines(){
		return lines;
	}
	
	public void tLineReset() {
		this.tLines=new ArrayList<Point>();
	}
	public String toText(){
		String s = "";
		for (shapes shape: DrawingCommands){
			s+=shape.toText()+"\n";
		}
		return s;
		
	}
	public void save() {
		String path = null;
		
		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		
		FileSystemView fsv = FileSystemView.getFileSystemView();
		System.out.println(fsv.getHomeDirectory());
		fileChooser.setDialogTitle("save");
		int returnVal = fileChooser.showOpenDialog(fileChooser);

		if(returnVal == JFileChooser.APPROVE_OPTION){ 
		path= fileChooser.getSelectedFile().getAbsolutePath();
		System.out.println(path);
		File file = new File(path+"/SaveDataPaint.txt");
		try {
			file.createNewFile();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
            FileWriter fileWritter = new FileWriter(path+"/SaveDataPaint.txt", true);    
            fileWritter.write(this.toText());
            fileWritter.close();  
            
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}  

		} 
	}
	public void load(){
		String path = null;
		
		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
		
		FileSystemView fsv = FileSystemView.getFileSystemView();
		System.out.println(fsv.getHomeDirectory());
		int returnVal = fileChooser.showOpenDialog(fileChooser);

		if(returnVal == JFileChooser.APPROVE_OPTION){ 
		path= fileChooser.getSelectedFile().getAbsolutePath();
		System.out.println(path);
		} 
	}
	public ArrayList<Point> getTLines(){
		return tLines;
	}
	
	public boolean polylineStatus() {
		return !(lines.size() % 2 == 0);
	}
	
	public void addCommand(shapes s) {
		this.DrawingCommands.push(s);
		this.Order.push(shapeToString(s));
		clearUndos();
	}
	
	private void clearUndos() {
		this.UndoCommands.clear();
		int orderSize = this.Order.size();
		Stack<String> Flip = new Stack<String>();
		for (int i = 0; i<orderSize; i++) {
			if(this.Order.peek().contains("Undo")) this.Order.pop();
			else Flip.push(this.Order.pop());
		}
		int flipSize = Flip.size();
		for (int i = 0; i < flipSize; i++)
			this.Order.push(Flip.pop());
	}

	public String shapeToString(shapes s) {
		int first = s.toString().indexOf("paint.")+6;
		int last = s.toString().indexOf("@");
		return s.toString().substring(first, last);
	}
	
	public void addTemp(shapes s) {
		this.tempDrawing.push(s);
		this.setChanged();
		this.notifyObservers();
	}
	
	public void emptyStacks() {
		this.DrawingCommands.clear();
		this.tempDrawing.clear();
		this.UndoCommands.clear();
		this.Order.clear();
		this.setChanged();
		this.notifyObservers();
	}
	
	public void undoCommand() {
		try {
			shapes s;
			s = this.DrawingCommands.pop();
			if(s.toString().contains("Invert")) {
				Invert("undo");
			}
			this.UndoCommands.push(s);
			
			String shape = shapeToString(s);
			ArrayList<String> Flip = new ArrayList<String>();
			while(this.Order.peek().contains("Undo")) {
				Flip.add(this.Order.pop()); 
			}
			this.Order.pop();
			this.Order.push("Undo "+shape);
			for (int i = Flip.size()-1; i>=0; i--) {
				this.Order.push(Flip.get(i));
			}
			
			this.setChanged();
			this.notifyObservers();
			}  
		catch (EmptyStackException e) {
			System.out.println("No commands to Undo");
		}
	}
	
	public void redoCommand() {
		try {
			System.out.println();
			shapes s = this.UndoCommands.pop();
			if(s.toString().contains("Invert")) {
				Invert("redo");
			}
			this.DrawingCommands.push(s);
			
			ArrayList<String> Flip = new ArrayList<String>();
			do { Flip.add(this.Order.pop()); }while(this.Order.peek().contains("Undo"));
			
			Flip.remove(0);
			this.Order.push(shapeToString(s));
			
			for (int i = Flip.size()-1; i>=0; i--) {
				this.Order.push(Flip.get(i));
			}
			
			this.setChanged();
			this.notifyObservers();
			}
		catch (EmptyStackException e) {
		}
	}

	public Stack<shapes> getDrawingCommands(){
		return DrawingCommands;
	}
	
	public Stack<shapes> getUndoCommands(){
		return UndoCommands;
	}
	
	public Stack<shapes> getTempDrawing(){
		return tempDrawing;
	}
	
	public Stack<String> getOrder(){
		return Order;
	}

	public void resetTemp() {
		this.tempDrawing =  new Stack<shapes>();
		
	}

	public void Invert(String r) {
		try {
		for(shapes s:DrawingCommands) {
			if(s != this.lastChanged) s.setColor(opposite(s.getColor()));
			this.lastChanged = s;
		}
		this.lastChanged = null;
		if (r!="undo")this.DrawingCommands.push(new Invert());
		else if(r == "undo")this.UndoCommands.push(new Invert());
		this.setChanged();
		this.notifyObservers();
		} catch (NullPointerException e) {
			
		}
	}
	
	public Color opposite(Color c) {
		int red = c.getRed();
		int green = c.getGreen();
		int blue = c.getBlue();
		return new Color(255-red,255-green,255-blue);
	}

	public void callHistory() {
		this.setChanged();
		this.notifyObservers();
	}

}
