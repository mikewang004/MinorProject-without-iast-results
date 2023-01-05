      Function SpreadingPressure(I,J,P)
      Implicit None

C     compute the spreading pressure for a given Pi0
      
      Include 'commons.inc'

      Integer I,J
      Double Precision P,SpreadingPressure



      If(Langmuir(I,J)) Then

         SpreadingPressure = Nimax(I,J)*Dlog(1.0d0+Ki(I,J)*P)

      Else

         SpreadingPressure = 
     &        Nimax(I,J)*Dlog(1.0d0+Ki(I,J)*(P**Pow(I,J)))/Pow(I,J)
         
      Endif

      
      Return
      End
