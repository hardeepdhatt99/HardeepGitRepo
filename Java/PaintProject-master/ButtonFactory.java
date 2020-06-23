package ca.utoronto.utm.paint;

public class ButtonFactory {
	   public Buttons getButton(String buttonType){
		      if(buttonType == null){
		         return null;
		      }        
		      if(buttonType.equalsIgnoreCase("ButtonImage")){
		         return new ButtonImage();
		      }
		      else {
		    	  return null;
		      }
	   }
}
