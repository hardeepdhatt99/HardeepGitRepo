package ca.utoronto.utm.paint;

import javax.swing.*;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.event.*;
import java.util.ArrayList;
/**
 * This is the top level View+Controller, it contains other aspects of the View+Controller.
 * @author arnold
 *
 */
public class View extends JFrame implements ActionListener {
	
	private static final long serialVersionUID = 1L;
	
	private PaintModel model;
	
	// The components that make this up
	private PaintPanel paintPanel;
	private ShapeChooserPanel shapeChooserPanel;
	private PanelColorAndThickness colorAndthickness = new PanelColorAndThickness();
	private HistoryPanel hPanel = new HistoryPanel(this);
	
	public View(PaintModel model) {
		super("Paint"); // set the title and do other JFrame init

		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setJMenuBar(createMenuBar());
		
		Container c=this.getContentPane();
	
		this.shapeChooserPanel = new ShapeChooserPanel(this);
		c.add(this.hPanel,BorderLayout.EAST);
		c.add(this.shapeChooserPanel,BorderLayout.WEST);
		
		c.add(this.colorAndthickness,BorderLayout.NORTH);
		this.model=model;		
		
		this.paintPanel = new PaintPanel(model, this);
		c.add(this.paintPanel, BorderLayout.CENTER);
		
		this.pack();
		this.setSize(1000,950);
		this.setVisible(true);
		
		colorAndthickness.getColorData().addObserver(paintPanel);
		colorAndthickness.getThicknessModel().addObserver(paintPanel);
		
		this.model.addObserver(hPanel);

		colorAndthickness.getTextBoxModel().addObserver(paintPanel);

	}

	public PaintPanel getPaintPanel(){
		return paintPanel;
	}
	
	public ShapeChooserPanel getShapeChooserPanel() {
		return shapeChooserPanel;
	}

	private JMenuBar createMenuBar() {
		JMenuBar menuBar = new JMenuBar();
		JMenu menu;
		JMenuItem menuItem;

		menu = new JMenu("File");

		// a group of JMenuItems
		menuItem = new JMenuItem("New");
		menuItem.addActionListener(this);
		menu.add(menuItem);

		menuItem = new JMenuItem("Open");
		menuItem.addActionListener(this);
		menu.add(menuItem);

		menuItem = new JMenuItem("Save");
		menuItem.addActionListener(this);
		menu.add(menuItem);

		menu.addSeparator();// -------------
	
		menuItem = new JMenuItem("Exit");
		menuItem.addActionListener(this);
		menu.add(menuItem);
		

		menuBar.add(menu);

		menu = new JMenu("Edit");

		// a group of JMenuItems
		menuItem = new JMenuItem("Cut");
		menuItem.addActionListener(this);
		menu.add(menuItem);

		menuItem = new JMenuItem("Copy");
		menuItem.addActionListener(this);
		menu.add(menuItem);

		menuItem = new JMenuItem("Paste");
		menuItem.addActionListener(this);
		menu.add(menuItem);
		
		menuItem = new JMenuItem("Clear");
		menuItem.addActionListener(this);
		menu.add(menuItem);
		
		menuItem = new JMenuItem("Invert");
		menuItem.addActionListener(this);
		menu.add(menuItem);
		
		menuItem = new JMenuItem("Change Background");
		menuItem.addActionListener(this);
		menu.add(menuItem);

		menu.addSeparator();// -------------

		menuItem = new JMenuItem("Undo");
		menuItem.addActionListener(this);
		menu.add(menuItem);

		menuItem = new JMenuItem("Redo");
		menuItem.addActionListener(this);
		menu.add(menuItem);

		menuBar.add(menu);

		return menuBar;
	}

	public void actionPerformed(ActionEvent e) {
		if(e.getActionCommand() == "Undo") {
			this.model.undoCommand();
		}else if(e.getActionCommand() == "Redo"){
			this.model.redoCommand();
		}else if(e.getActionCommand() == "Clear") {
			this.paintPanel.clear();
		} else if(e.getActionCommand() == "Exit") {
			System.exit(0);
		} else if(e.getActionCommand() == "Save") {
			this.model.save();
		} else if(e.getActionCommand() == "Open") {
			this.model.load();
		} else if(e.getActionCommand() == "Invert") {
			this.model.Invert("");
		} else if(e.getActionCommand() == "Change Background") {
			this.paintPanel.setBackground(this.paintPanel.getColor());
		}
		
	}
	
}
